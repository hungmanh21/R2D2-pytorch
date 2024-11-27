"""
PyTorch implementation of R2D2 (Reinforcement Learning with Debate)
Original TensorFlow implementation adapted to PyTorch while maintaining functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher

logger = logging.getLogger()


class Episode(nn.Module):
    """
    Class representing an episode for reinforcement learning.
    Converted from TensorFlow to PyTorch implementation.
    """

    def __init__(self, graph, data, params):
        """
        Initializes an episode.
        
        Args:
            graph: RelationEntityGrapher object
            data: Tuple containing (start_entities, query_relation, end_entities, labels, all_answers)
            params: Tuple containing (num_rollouts, mode)
        """
        super(Episode, self).__init__()
        self.grapher = graph
        self.num_rollouts, self.mode = params
        self.current_hop = 0

        start_entities, query_relation, end_entities, labels, all_answers = data
        self.no_examples = start_entities.shape[0]

        # Convert numpy arrays to PyTorch tensors
        start_entities = torch.from_numpy(
            np.repeat(start_entities, self.num_rollouts))
        batch_query_relation = torch.from_numpy(
            np.repeat(query_relation, self.num_rollouts))
        end_entities = torch.from_numpy(
            np.repeat(end_entities, self.num_rollouts))

        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = start_entities.clone()
        self.query_relation = batch_query_relation
        self.all_answers = all_answers
        self.labels = torch.from_numpy(np.repeat(labels, self.num_rollouts))

        # Get next actions from grapher
        next_actions = self.grapher.return_next_actions(
            self.current_entities.numpy(),
            self.start_entities.numpy(),
            self.query_relation.numpy(),
            self.end_entities.numpy(),
            self.labels.numpy(),
            self.all_answers,
            self.num_rollouts
        )

        # Convert to PyTorch tensors and store state
        next_actions = torch.from_numpy(next_actions)
        self.state = {
            'next_relations': next_actions[:, :, 1],
            'next_entities': torch.where(
                next_actions[:, :, 0] == self.start_entities.unsqueeze(
                    -1).expand(-1, next_actions.shape[1]),
                torch.tensor(self.grapher.get_placeholder_subject()),
                next_actions[:, :, 0]
            ),
            'current_entities': torch.where(
                self.current_entities == self.start_entities,
                torch.tensor(self.grapher.get_placeholder_subject()),
                self.current_entities
            )
        }

        self.init_state = dict(self.state)

    def reset_initial_state(self) -> Dict:
        """Returns the initial state of the episode."""
        self.state = dict(self.init_state)
        return self.state

    def get_query_relation(self) -> torch.Tensor:
        """Returns the query relations for the episode."""
        return self.query_relation

    def get_default_query_subject(self) -> str:
        """Returns the placeholder string for query subject."""
        return self.grapher.QUERY_SUBJECT_NAME

    def get_query_subjects(self) -> torch.Tensor:
        """Returns the query subjects."""
        return self.start_entities

    def get_query_objects(self) -> torch.Tensor:
        """Returns the query objects."""
        return self.end_entities

    def get_labels(self) -> torch.Tensor:
        """Returns the query labels reshaped to [batch_size, 1]."""
        return self.labels.unsqueeze(1)

    def get_rewards(self, logits_sequence: List[Tuple[int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes rewards for the agents based on logits sequence.
        
        Args:
            logits_sequence: List of tuples (agent_id, logit_value)
            
        Returns:
            Tuple of rewards for both agents
        """
        rewards_1 = []
        rewards_2 = []

        for which_agent, logit in logits_sequence:
            if not which_agent:
                rewards_1.append(logit)
            else:
                rewards_2.append(logit)

        rewards_1 = torch.stack(rewards_1, dim=1)
        rewards_2 = torch.stack(rewards_2, dim=1)

        rewards_1 = rewards_1.squeeze(-1)
        rewards_2 = -rewards_2.squeeze(-1)

        return rewards_1, rewards_2

    def __call__(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes a transition in the environment based on action.
        
        Args:
            action: Tensor of actions to take
            
        Returns:
            Updated state dictionary
        """
        self.current_hop += 1

        # Handle next entities selection
        true_next_entities = torch.where(
            self.state['next_entities'] == self.grapher.get_placeholder_subject(),
            self.start_entities.unsqueeze(-1).expand(-1,
                                                     self.state['next_entities'].shape[1]),
            self.state['next_entities']
        )

        # Update current entities based on action
        batch_indices = torch.arange(self.no_examples * self.num_rollouts)
        self.current_entities = true_next_entities[batch_indices, action]

        # Get next actions from grapher
        next_actions = self.grapher.return_next_actions(
            self.current_entities.numpy(),
            self.start_entities.numpy(),
            self.query_relation.numpy(),
            self.end_entities.numpy(),
            self.labels.numpy(),
            self.all_answers,
            self.num_rollouts
        )

        next_actions = torch.from_numpy(next_actions)

        # Update state
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = torch.where(
            next_actions[:, :, 0] == self.start_entities.unsqueeze(
                -1).expand(-1, next_actions.shape[1]),
            torch.tensor(self.grapher.get_placeholder_subject()),
            next_actions[:, :, 0]
        )
        self.state['current_entities'] = torch.where(
            self.current_entities == self.start_entities,
            torch.tensor(self.grapher.get_placeholder_subject()),
            self.current_entities
        )

        return self.state


class Environment:
    """
    PyTorch implementation of the R2D2 environment.
    Handles episode generation and management.
    """

    def __init__(self, params: Dict, mode: str = 'train'):
        """
        Initializes the environment.
        
        Args:
            params: Dictionary of parameters
            mode: Either 'train', 'test' or 'dev'
        """
        self.num_rollouts = params['num_rollouts']
        self.mode = mode
        self.test_rollouts = params['test_rollouts']
        self.rounds_sub_training = params['rounds_sub_training']
        input_dir = params['data_input_dir']

        # Initialize batcher based on mode
        if mode == 'train':
            self.batcher = RelationEntityBatcher(
                input_dir=input_dir,
                batch_size=params['batch_size'],
                entity_vocab=params['entity_vocab'],
                relation_vocab=params['relation_vocab'],
                is_use_fixed_false_facts=params['is_use_fixed_false_facts'],
                num_false_facts=params['false_facts_train'],
                rounds_sub_training=self.rounds_sub_training
            )
        else:
            self.batcher = RelationEntityBatcher(
                input_dir=input_dir,
                batch_size=params['batch_size'],
                entity_vocab=params['entity_vocab'],
                relation_vocab=params['relation_vocab'],
                is_use_fixed_false_facts=params['is_use_fixed_false_facts'],
                mode=mode
            )

        # Initialize grapher
        self.grapher = RelationEntityGrapher(
            data_input_dir=params['data_input_dir'],
            mode=mode,
            max_num_actions=params['max_num_actions'],
            entity_vocab=params['entity_vocab'],
            relation_vocab=params['relation_vocab'],
            mid_to_name=params['mid_to_name']
        )

    def get_episodes(self):
        """
        Generator that yields episodes.
        
        Yields:
            Episode objects for either training or testing
        """
        if self.mode == 'train':
            params = (self.num_rollouts, self.mode)
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            params = (self.test_rollouts, self.mode)
            for data in self.batcher.yield_next_batch_test():
                if data is None:
                    return
                yield Episode(self.grapher, data, params)

import torch
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

logger = logging.getLogger()


class Episode:
    """
    Class representing an episode for reinforcement learning.
    """

    def __init__(self, graph, data, params):
        """
        Initializes an episode.

        :param graph: RelationEntityGrapher. Graph for the episode defining the available actions.
        :param data: Tuple. Contains query subjects, relations, objects, labels, and all correct answers.
        :param params: Tuple. Number of rollouts and mode ('train', 'test', or 'dev').
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.grapher = graph
        num_rollouts, mode = params
        self.mode = mode
        self.num_rollouts = num_rollouts
        self.current_hop = 0

        start_entities, query_relation, end_entities, labels, all_answers = data
        self.no_examples = start_entities.shape[0]

        start_entities = np.repeat(start_entities, self.num_rollouts)
        query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        labels = np.repeat(labels, self.num_rollouts)

        self.start_entities = torch.tensor(
            start_entities, dtype=torch.long, device=self.device)
        self.end_entities = torch.tensor(
            end_entities, dtype=torch.long, device=self.device)
        self.current_entities = self.start_entities.clone()
        self.query_relation = torch.tensor(
            query_relation, dtype=torch.long, device=self.device)
        self.labels = torch.tensor(
            labels, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.all_answers = all_answers  # Keep as NumPy if needed by grapher

        next_actions = self.grapher.return_next_actions(
            self.current_entities.cpu().numpy(),
            self.start_entities.cpu().numpy(),
            self.query_relation.cpu().numpy(),
            self.end_entities.cpu().numpy(),
            self.labels.cpu().numpy(),
            self.all_answers,
            self.num_rollouts,
        )

        self.state = self._process_state(next_actions)
        self.init_state = dict(self.state)

    def _process_state(self, next_actions):
        """
        Processes the next actions and prepares the episode state.

        :param next_actions: NumPy array of next actions.
        :return: Dictionary with next relations, next entities, and current entities.
        """
        next_relations = torch.tensor(
            next_actions[:, :, 1], dtype=torch.long, device=self.device)
        next_entities = torch.tensor(
            next_actions[:, :, 0], dtype=torch.long, device=self.device)

        start_mask = torch.tensor(
            next_actions[:, :, 0] == np.expand_dims(
                self.start_entities.cpu().numpy(), axis=-1),
            dtype=torch.bool,
            device=self.device,
        )
        current_mask = self.current_entities == self.start_entities

        state = {
            "next_relations": next_relations,
            "next_entities": torch.where(
                start_mask, self.grapher.get_placeholder_subject(), next_entities
            ),
            "current_entities": torch.where(
                current_mask, self.grapher.get_placeholder_subject(), self.current_entities
            ),
        }
        return state

    def reset_initial_state(self):
        """
        Resets the state to the initial configuration.
        """
        self.state = dict(self.init_state)
        return self.state

    def get_query_relation(self):
        """
        Getter for the query's relations for the episode.
        """
        return self.query_relation

    def get_query_subjects(self):
        """
        Getter for the query's subjects of the episode.
        """
        return self.start_entities

    def get_query_objects(self):
        """
        Getter for the query's objects of the episode.
        """
        return self.end_entities

    def get_labels(self):
        """
        Getter for the query's labels of the episode.
        """
        return self.labels

    def get_rewards(self, logits_sequence):
        """
        Computes and returns rewards for the episode.

        :param logits_sequence: List of tuples with logits for agent actions.
        :return: Tuple of rewards for agents.
        """
        rewards_1, rewards_2 = [], []
        for which_agent, logit in logits_sequence:
            if which_agent == 0:
                rewards_1.append(logit)
            else:
                rewards_2.append(logit)

        rewards_1 = torch.stack(rewards_1, dim=1).squeeze(-1)
        rewards_2 = -torch.stack(rewards_2, dim=1).squeeze(-1)

        return rewards_1, rewards_2

    def __call__(self, action):
        """
        Simulates a transition on the graph defined by the action.

        :param action: Torch tensor, [batch_size]. Action indices.
        :return: Updated state of the episode.
        """
        self.current_hop += 1
        true_next_entities = torch.where(
            self.state["next_entities"] == self.grapher.get_placeholder_subject(),
            self.start_entities.unsqueeze(-1),
            self.state["next_entities"],
        )

        self.current_entities = true_next_entities[
            torch.arange(self.no_examples * self.num_rollouts,
                         device=self.device), action
        ]

        next_actions = self.grapher.return_next_actions(
            self.current_entities.cpu().numpy(),
            self.start_entities.cpu().numpy(),
            self.query_relation.cpu().numpy(),
            self.end_entities.cpu().numpy(),
            self.labels.cpu().numpy(),
            self.all_answers,
            self.num_rollouts,
        )

        self.state = self._process_state(next_actions)
        return self.state


class env:
    """
    Class representing an environment for the model.
    """

    def __init__(self, params, mode="train"):
        """
        Initializes the environment.

        :param params: Dictionary of experiment parameters.
        :param mode: Mode ('train', 'test', or 'dev') for the environment.
        """
        self.num_rollouts = params["num_rollouts"]
        self.mode = mode
        self.test_rollouts = params["test_rollouts"]
        self.rounds_sub_training = params["rounds_sub_training"]

        input_dir = params["data_input_dir"]
        self.batcher = RelationEntityBatcher(
            input_dir=input_dir,
            batch_size=params["batch_size"],
            entity_vocab=params["entity_vocab"],
            relation_vocab=params["relation_vocab"],
            is_use_fixed_false_facts=params["is_use_fixed_false_facts"],
            num_false_facts=params["false_facts_train"] if mode == "train" else 0,
            mode=mode,
        )

        self.grapher = RelationEntityGrapher(
            data_input_dir=params["data_input_dir"],
            mode=mode,
            max_num_actions=params["max_num_actions"],
            entity_vocab=params["entity_vocab"],
            relation_vocab=params["relation_vocab"],
            mid_to_name=params["mid_to_name"],
        )

    def get_episodes(self):
        """
        Yields episodes from the environment.
        """
        params = (self.num_rollouts, self.mode) if self.mode == "train" else (
            self.test_rollouts, self.mode)

        for data in (
            self.batcher.yield_next_batch_train()
            if self.mode == "train"
            else self.batcher.yield_next_batch_test()
        ):
            if data is None:
                return
            yield Episode(self.grapher, data, params)

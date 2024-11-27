from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Agent(nn.Module):
    """
    PyTorch implementation of the Agent module.
    Represents both pro and con agents in the debate.
    """

    def __init__(self, params: Dict, judge):
        super(Agent, self).__init__()
        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.test_rollouts = params['test_rollouts']
        self.path_length = params['path_length']
        self.batch_size = params['batch_size'] * \
            (1 + params['false_facts_train']) * params['num_rollouts']
        self.use_entity_embeddings = params['use_entity_embeddings']

        # Define embeddings for both agents
        self.relation_embeddings_agent1 = nn.Embedding(
            self.action_vocab_size, self.embedding_size)
        self.entity_embeddings_agent1 = nn.Embedding(
            self.entity_vocab_size, self.embedding_size)
        self.relation_embeddings_agent2 = nn.Embedding(
            self.action_vocab_size, self.embedding_size)
        self.entity_embeddings_agent2 = nn.Embedding(
            self.entity_vocab_size, self.embedding_size)

        # Input size depends on whether we use entity embeddings
        lstm_input_size = (
            2 if self.use_entity_embeddings else 1) * self.embedding_size
        lstm_input_size += 3 * self.embedding_size  # For query information

        # Create LSTMs for both agents
        self.policy_agent1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=params['layers_agent'],
            batch_first=True
        )

        self.policy_agent2 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=params['layers_agent'],
            batch_first=True
        )

        # Create MLPs for action selection
        self.action_selector1 = nn.Linear(
            self.hidden_size, self.embedding_size)
        self.action_selector2 = nn.Linear(
            self.hidden_size, self.embedding_size)

    def get_init_state_array(self, temp_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns initial hidden states for both agents' LSTMs."""
        h1 = torch.zeros(self.policy_agent1.num_layers,
                         temp_batch_size * self.test_rollouts, self.hidden_size)
        c1 = torch.zeros(self.policy_agent1.num_layers,
                         temp_batch_size * self.test_rollouts, self.hidden_size)

        h2 = torch.zeros(self.policy_agent2.num_layers,
                         temp_batch_size * self.test_rollouts, self.hidden_size)
        c2 = torch.zeros(self.policy_agent2.num_layers,
                         temp_batch_size * self.test_rollouts, self.hidden_size)

        return (h1, c1), (h2, c2)

    def set_query_embeddings(self, query_subject: torch.Tensor, query_relation: torch.Tensor,
                             query_object: torch.Tensor):
        """Gets and stores query embeddings for both agents."""
        # Agent 1 embeddings
        self.query_subject_embedding_agent1 = self.entity_embeddings_agent1(
            query_subject)
        self.query_relation_embedding_agent1 = self.relation_embeddings_agent1(
            query_relation)
        self.query_object_embedding_agent1 = self.entity_embeddings_agent1(
            query_object)

        # Agent 2 embeddings
        self.query_subject_embedding_agent2 = self.entity_embeddings_agent2(
            query_subject)
        self.query_relation_embedding_agent2 = self.relation_embeddings_agent2(
            query_relation)
        self.query_object_embedding_agent2 = self.entity_embeddings_agent2(
            query_object)

    def action_encoder_agent(self, next_relations: torch.Tensor, next_entities: torch.Tensor,
                             which_agent: torch.Tensor) -> torch.Tensor:
        """Encodes available actions for an agent."""
        # Select embeddings based on which agent
        relation_embeddings = torch.where(which_agent == 0,
                                          self.relation_embeddings_agent1(
                                              next_relations),
                                          self.relation_embeddings_agent2(next_relations))

        if self.use_entity_embeddings:
            entity_embeddings = torch.where(which_agent == 0,
                                            self.entity_embeddings_agent1(
                                                next_entities),
                                            self.entity_embeddings_agent2(next_entities))
            return torch.cat([relation_embeddings, entity_embeddings], dim=-1)

        return relation_embeddings

    def step(self, next_relations: torch.Tensor, next_entities: torch.Tensor,
             prev_state_agent1: Tuple[torch.Tensor, torch.Tensor],
             prev_state_agent2: Tuple[torch.Tensor, torch.Tensor],
             prev_relation: torch.Tensor, current_entities: torch.Tensor,
             range_arr: torch.Tensor, which_agent: torch.Tensor,
             random_flag: torch.Tensor) -> Tuple:
        """Executes one step in the debate for an agent."""
        # Get state representation
        if which_agent == 0:
            prev_entity = self.entity_embeddings_agent1(current_entities)
            prev_relation_emb = self.relation_embeddings_agent1(prev_relation)
            query_subject = self.query_subject_embedding_agent1
            query_relation = self.query_relation_embedding_agent1
            query_object = self.query_object_embedding_agent1
            policy = self.policy_agent1
            action_selector = self.action_selector1
            prev_state = prev_state_agent1
        else:
            prev_entity = self.entity_embeddings_agent2(current_entities)
            prev_relation_emb = self.relation_embeddings_agent2(prev_relation)
            query_subject = self.query_subject_embedding_agent2
            query_relation = self.query_relation_embedding_agent2
            query_object = self.query_object_embedding_agent2
            policy = self.policy_agent2
            action_selector = self.action_selector2
            prev_state = prev_state_agent2

        # Create state input
        if self.use_entity_embeddings:
            state = torch.cat([prev_relation_emb, prev_entity], dim=-1)
        else:
            state = prev_relation_emb

        state = torch.cat([
            state, query_subject, query_relation, query_object
        ], dim=-1)

        # Pass through LSTM
        state = state.unsqueeze(1)  # Add sequence dimension
        output, new_state = policy(state, prev_state)
        output = output.squeeze(1)  # Remove sequence dimension

        # Get action scores
        action_query = action_selector(output)
        candidate_actions = self.action_encoder_agent(
            next_relations, next_entities, which_agent)
        scores = torch.bmm(candidate_actions,
                           action_query.unsqueeze(-1)).squeeze(-1)

        # Mask padding actions
        # PAD token
        mask = (next_relations == self.judge.relation_embeddings.num_embeddings - 1)
        scores = torch.where(mask, torch.full_like(
            scores, float('-inf')), scores)

        # Sample action
        if random_flag:
            probs = torch.ones_like(scores)
            probs = torch.where(mask, torch.zeros_like(probs), probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            action = torch.multinomial(probs, 1)
        else:
            action = torch.multinomial(F.softmax(scores, dim=-1), 1)

        action_idx = action.squeeze(-1)

        # Compute loss
        loss = F.cross_entropy(scores, action_idx)

        # Get chosen relation
        chosen_relation = next_relations[torch.arange(
            next_relations.size(0)), action_idx]

        # Update states
        if which_agent == 0:
            new_state_agent1 = new_state
            new_state_agent2 = prev_state_agent2
        else:
            new_state_agent1 = prev_state_agent1
            new_state_agent2 = new_state

        return loss, new_state_agent1, new_state_agent2, F.log_softmax(scores, dim=-1), action_idx, chosen_relation

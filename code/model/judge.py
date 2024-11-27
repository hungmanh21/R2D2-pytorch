import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class Judge(nn.Module):
    """
    PyTorch implementation of the Judge module.
    Evaluates arguments and provides feedback for the debate.
    """

    def __init__(self, params: Dict):
        super(Judge, self).__init__()
        self.batch_size = params['batch_size'] * \
            (1 + params['false_facts_train']) * params['num_rollouts']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.path_length = params['path_length']
        self.train_relations = params['train_relation_embeddings']
        self.train_entities = params['train_entity_embeddings']
        
        # Create embeddings
        self.relation_embeddings = nn.Embedding(
            self.action_vocab_size, self.embedding_size)
        self.entity_embeddings = nn.Embedding(
            self.entity_vocab_size, self.embedding_size)

        # Create argument encoder LSTM
        self.argument_encoder = nn.LSTM(
            input_size=2 * self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Create MLPs for classification
        self.argument_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def set_labels_placeholder(self, labels: torch.Tensor):
        """Sets the labels tensor."""
        self.labels = labels

    def set_query_embeddings(self, query_subject: torch.Tensor, query_relation: torch.Tensor,
                             query_object: torch.Tensor):
        """Gets and stores query embeddings."""
        self.query_subject_embedding = self.entity_embeddings(query_subject)
        self.query_relation_embedding = self.relation_embeddings(
            query_relation)
        self.query_object_embedding = self.entity_embeddings(query_object)

    def action_encoder_judge(self, prev_relation: torch.Tensor, prev_entity: torch.Tensor) -> torch.Tensor:
        """Encodes actions for the judge."""
        relation_embedding = self.relation_embeddings(prev_relation)
        entity_embedding = self.entity_embeddings(prev_entity)
        return torch.cat([relation_embedding, entity_embedding], dim=-1)

    def extend_argument(self, argument: torch.Tensor, t: torch.Tensor, action_idx: torch.Tensor,
                        next_relations: torch.Tensor, next_entities: torch.Tensor,
                        range_arr: torch.Tensor) -> torch.Tensor:
        """Extends an argument with new actions."""
        batch_indices = torch.arange(argument.size(0), device=argument.device)
        chosen_relation = next_relations[batch_indices, action_idx]
        chosen_entity = next_entities[batch_indices, action_idx]

        new_action = self.action_encoder_judge(chosen_relation, chosen_entity)

        if t == 0:
            return new_action.unsqueeze(1)
        else:
            return torch.cat([argument, new_action.unsqueeze(1)], dim=1)

    def classify_argument(self, argument: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classifies a single argument."""
        # Process through LSTM
        output, (hidden, _) = self.argument_encoder(argument)
        hidden = hidden.squeeze(0)

        # Get logits
        logits = self.argument_classifier(hidden)
        return logits, hidden

    def get_logits_argument(self, hidden_rep: torch.Tensor) -> torch.Tensor:
        """Gets logits from hidden representation."""
        return self.final_classifier(hidden_rep)

    def final_loss(self, argument_representations: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes final loss based on all arguments."""
        mean_representation = torch.stack(
            argument_representations, dim=-1).mean(dim=-1)
        final_logits = self.final_classifier(mean_representation)
        loss = F.binary_cross_entropy_with_logits(
            final_logits, self.labels.float())
        return loss, final_logits

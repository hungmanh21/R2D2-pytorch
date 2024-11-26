import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Judge(nn.Module):
    def __init__(self, params):
        """
        Initializes the judge.

        :param params: Dict. Parameters of the experiment.
        """
        super().__init__()

        # Basic parameters
        self.path_length = params['path_length']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.use_entity_embeddings = params.get('use_entity_embeddings', False)
        self.train_entities = params.get('train_entity_embeddings', True)
        self.train_relations = params.get('train_relation_embeddings', True)
        self.hidden_layers = params.get('layers_judge', 2)

        # Embeddings
        self.relation_embedding = nn.Embedding(
            self.action_vocab_size,
            self.embedding_size,
            _weight=self._get_xavier_initialization(
                self.action_vocab_size, self.embedding_size)
        )
        self.relation_embedding.weight.requires_grad = True if self.train_relations == 1 else False

        # Entity embedding initialization
        if self.use_entity_embeddings:
            self.entity_embedding = nn.Embedding(
                self.entity_vocab_size,
                self.embedding_size,
                _weight=self._get_xavier_initialization(
                    self.entity_vocab_size, self.embedding_size)
            )
        else:
            self.entity_embedding = nn.Embedding(
                self.entity_vocab_size,
                self.embedding_size,
                _weight=torch.zeros(self.entity_vocab_size,
                                    self.embedding_size)
            )
        self.entity_embedding.weight.requires_grad = True if self.train_entities == 1 else False

        # Classifier MLP
        mlp_layers = []
        input_dim = (self.path_length *
                     (2 if self.use_entity_embeddings else 1) + 2) * self.embedding_size
        for i in range(self.hidden_layers - 1):
            mlp_layers.append(nn.Linear(input_dim if i ==
                              0 else self.hidden_size, self.hidden_size))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.mlp = nn.Sequential(*mlp_layers)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        # Placeholders (using attributes instead)
        self.query_subject_embedding = None
        self.query_relation_embedding = None
        self.query_object_embedding = None
        self.labels = None

    def _get_xavier_initialization(self, num_rows, embedding_size):
        """
        Create xavier initialized weights
        """
        weights = torch.empty(num_rows, embedding_size)
        nn.init.xavier_uniform_(weights)
        return weights

    def action_encoder_judge(self, next_relations, next_entities):
        """
        Encodes an action into its embedded representation.
        """
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)

        if self.use_entity_embeddings:
            return torch.cat([relation_embedding, entity_embedding], dim=-1)
        return relation_embedding

    def extend_argument(self, argument, t, action_idx, next_relations, next_entities, range_arr):
        """
        Extends an argument by adding the embedded representation of an action.
        """
        # Get chosen relation and entity using advanced indexing
        chosen_relation = next_relations[range_arr, action_idx]
        chosen_entities = next_entities[range_arr, action_idx]

        # Encode the action
        action_embedding = self.action_encoder_judge(
            chosen_relation, chosen_entities)

        # Extend or replace argument based on step
        if t % self.path_length == 0:
            return action_embedding
        else:
            return torch.cat([argument, action_embedding], dim=-1)

    def classify_argument(self, argument):
        """
        Classifies arguments by computing a hidden representation and assigning logits.
        """
        # Concatenate query embeddings
        argument = torch.cat([
            argument,
            self.query_relation_embedding,
            self.query_object_embedding
        ], dim=-1)

        # Reshape argument for MLP
        if self.use_entity_embeddings:
            argument = argument.view(-1, self.path_length *
                                     2 * self.embedding_size + 2 * self.embedding_size)
        else:
            argument = argument.view(-1, self.path_length *
                                     self.embedding_size + 2 * self.embedding_size)

        # Pass through MLP
        hidden = self.mlp(argument)

        # Get logits
        logits = self.classifier(hidden)

        return logits, hidden

    def final_loss(self, rep_argu_list):
        """
        Computes the final loss and final logits of the debates using all arguments presented.
        """
        # Average the arguments
        average_argu = torch.mean(torch.stack(rep_argu_list), dim=0)

        # Get final logit
        final_logit = self.classifier(average_argu)

        # Compute loss
        final_loss = F.binary_cross_entropy_with_logits(
            final_logit,
            self.labels.float()
        )

        return final_loss, final_logit

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        """
        Sets the judge's query embeddings.
        """
        self.query_subject_embedding = self.entity_embedding(query_subject)
        self.query_relation_embedding = self.relation_embedding(query_relation)
        self.query_object_embedding = self.entity_embedding(query_object)

    def set_labels_placeholder(self, labels):
        """
        Setter for the labels.
        """
        self.labels = labels

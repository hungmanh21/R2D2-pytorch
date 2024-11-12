import torch
import torch.nn as nn
import torch.nn.functional as F

class Judge(nn.Module):
    """
    Class representing the Judge in R2D2. Adapted from the agent class from MINERVA.
    Evaluates the arguments that the agents present and assigns them a score.
    """

    def __init__(self, params):
        """
        Initializes the Judge.

        :param params: Dict. Parameters of the experiment.
        """
        super(Judge, self).__init__()

        self.path_length = params['path_length']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.use_entity_embeddings = params['use_entity_embeddings']
        self.hidden_layers = params['layers_judge']
        self.batch_size = params['batch_size'] * (1 + params['false_facts_train']) * params['num_rollouts']

        self.relation_embedding = nn.Embedding(self.action_vocab_size, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, self.embedding_size)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        if self.use_entity_embeddings:
            nn.init.xavier_uniform_(self.entity_embedding.weight)

        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.relation_embedding.weight.requires_grad = self.train_relations
        self.entity_embedding.weight.requires_grad = self.train_entities

        # Define MLP layers for classification
        self.mlp_layers = nn.ModuleList()
        for i in range(self.hidden_layers - 1):
            self.mlp_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def action_encoder_judge(self, next_relations, next_entities):
        """
        Encodes an action into its embedded representation.
        """
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)

        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    def extend_argument(self, argument, t, action_idx, next_relations, next_entities, range_arr):
        """
        Extends an argument by adding the embedded representation of an action.
        """
        chosen_relation = next_relations[range_arr, action_idx]
        chosen_entity = next_entities[range_arr, action_idx]
        action_embedding = self.action_encoder_judge(chosen_relation, chosen_entity)

        if t % self.path_length == 0:
            argument = action_embedding
        else:
            argument = torch.cat([argument, action_embedding], dim=-1)

        return argument

    def classify_argument(self, argument, query_relation_embedding, query_object_embedding):
        """
        Classifies arguments by computing a hidden representation and assigning logits.
        """
        argument = torch.cat([argument, query_relation_embedding, query_object_embedding], dim=-1)

        if self.use_entity_embeddings:
            argument_dim = self.path_length * 2 * self.embedding_size + 2 * self.embedding_size
        else:
            argument_dim = self.path_length * self.embedding_size + 2 * self.embedding_size

        argument = argument.view(-1, argument_dim)
        hidden = argument

        for layer in self.mlp_layers:
            hidden = F.relu(layer(hidden))
        logits = self.output_layer(hidden)

        return logits, hidden

    def final_loss(self, rep_argu_list, labels):
        """
        Computes the final loss and logits for the debates using all arguments.
        """
        average_argu = torch.mean(torch.stack(rep_argu_list, dim=-1), dim=-1)
        logits = self.output_layer(average_argu)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss, logits

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        """
        Sets the judge's query information.
        """
        self.query_subject_embedding = self.entity_embedding(query_subject)
        self.query_relation_embedding = self.relation_embedding(query_relation)
        self.query_object_embedding = self.entity_embedding(query_object)

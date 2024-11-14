import torch
import torch.nn as nn
import torch.nn.functional as F


class Judge(nn.Module):
    """
    Class representing the Judge in R2D2. Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.

    It evaluates the arguments that the agents present and assigns them a score
    that is used to train the agents. Furthermore, it assigns the final prediction score to the whole debate.
    """

    def __init__(self, params):
        """
        Initializes the judge.

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

        # Embedding layers
        self.relation_embeddings = nn.Embedding(
            self.action_vocab_size, self.embedding_size
        )
        self.entity_embeddings = nn.Embedding(
            self.entity_vocab_size, self.embedding_size
        )

        if params['use_entity_embeddings']:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        else:
            nn.init.zeros_(self.entity_embeddings.weight)

        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.entity_embeddings.weight.requires_grad = True if self.train_entities else False
        self.relation_embeddings.weight.requires_grad = True if self.train_relations else False

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Define MLP layers
        self.mlp = nn.ModuleList()
        for i in range(self.hidden_layers):
            input_dim = (
                self.hidden_size
                if i > 0
                else (
                    self.path_length * 2 * self.embedding_size
                    + 2 * self.embedding_size
                    if self.use_entity_embeddings
                    else self.path_length * self.embedding_size + 2 * self.embedding_size
                )
            )
            self.mlp.append(nn.Linear(input_dim, self.hidden_size))

        # Output layer
        self.classifier = nn.Linear(self.hidden_size, 1)

    def action_encoder_judge(self, next_relations, next_entities):
        """
        Encodes an action into its embedded representation.

        :param next_relations: Tensor, [Batch_size]. Tensor with the ids of the picked relations.
        :param next_entities: Tensor, [Batch_size]. Tensor with the ids of the picked target entities.
        :return: Tensor. Embedded representation of the picked action.
        """
        relation_embedding = self.relation_embeddings(next_relations)
        entity_embedding = self.entity_embeddings(next_entities)
        if self.use_entity_embeddings:
            action_embedding = torch.cat(
                [relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding
        return action_embedding

    def extend_argument(self, argument, t, action_idx, next_relations, next_entities, range_arr):
        """
        Extends an argument by adding the embedded representation of an action.

        :param argument: Tensor, [Batch_size, None]. Argument to be extended.
        :param t: Tensor, []. Step number for the episode.
        :param action_idx: Tensor, [Batch_size]. Number of the selected action.
        :param next_relations: Tensor, [Batch_size, max_num_actions]. Contains the ids of all possible next relations.
        :param next_entities: Tensor, [Batch_size, max_num_actions]. Contains the ids of all possible next entities.
        :param range_arr: Tensor, [Batch_size]. Range tensor to select the correct next action.
        :return: Tensor. Extended argument.
        """
        chosen_relation = next_relations[range_arr, action_idx]
        chosen_entity = next_entities[range_arr, action_idx]
        action_embedding = self.action_encoder_judge(
            chosen_relation, chosen_entity)

        if t % self.path_length == 0:
            argument = action_embedding
        else:
            argument = torch.cat([argument, action_embedding], dim=-1)

        return argument

    def classify_argument(self, argument, query_relation_embedding, query_object_embedding):
        """
        Classifies arguments by computing a hidden representation and assigning logits.

        :param argument: Tensor. Embedded representation of the arguments.
        :param query_relation_embedding: Tensor. Embedded query relation.
        :param query_object_embedding: Tensor. Embedded query object.
        :return: Tensor. Logits for the arguments and hidden representation.
        """
        argument = torch.cat(
            [argument, query_relation_embedding, query_object_embedding], dim=-1
        )
        hidden = argument
        for layer in self.mlp:
            hidden = F.relu(layer(hidden))
        logits = self.classifier(hidden)
        return logits, hidden

    def final_loss(self, rep_argu_list, labels):
        """
        Computes the final loss and final logits of the debates using all arguments presented.

        :param rep_argu_list: List of Tensors. Hidden representations of the arguments.
        :param labels: Tensor. Ground truth labels.
        :return: Tuple. Final loss and final logits.
        """
        average_argu = torch.mean(torch.stack(rep_argu_list, dim=-1), dim=-1)
        logits = self.classifier(average_argu)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        """
        Sets the judge's query information.
        """
        query_subject = torch.tensor(
            query_subject, dtype=torch.long, device=self.device)
        query_relation = torch.tensor(
            query_relation, dtype=torch.long, device=self.device)
        query_object = torch.tensor(
            query_object, dtype=torch.long, device=self.device)

        self.query_subject_embedding = self.entity_embeddings(query_subject)
        self.query_relation_embedding = self.relation_embeddings(
            query_relation)
        self.query_object_embedding = self.entity_embeddings(query_object)

import numpy as np
import torch
import torch.nn as nn


class Agent(nn.Module):
    '''
    Class for the agents in R2D2. Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.
    A single instance of Agent contains both the pro and con agent.
    '''

    def __init__(self, params, judge):
        super(Agent, self).__init__()
        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.ePAD = torch.tensor(
            params['entity_vocab']['PAD'], dtype=torch.int32)
        self.rPAD = torch.tensor(
            params['relation_vocab']['PAD'], dtype=torch.int32)
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.test_rollouts = params['test_rollouts']
        self.path_length = params['path_length']
        self.batch_size = params['batch_size'] * \
            (1 + params['false_facts_train']) * params['num_rollouts']
        self.dummy_start_label = torch.ones(
            self.batch_size, dtype=torch.int64) * params['relation_vocab']['DUMMY_START_RELATION']

        self.hidden_layers = params['layers_agent']
        self.custom_baseline = params['custom_baseline']
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.entity_initializer = nn.init.xavier_uniform_
            self.m = 2
        else:
            self.entity_initializer = nn.init.zeros_
            self.m = 1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.define_embeddings()
        self.define_agents_policy()

    def define_embeddings(self):
        '''
        For both agents, creates and adds the embeddings for the KG's relations and entities.
        '''
        # Agent 1 embeddings
        self.relation_lookup_table_agent_1 = nn.Embedding(
            self.action_vocab_size, self.embedding_size)
        self.entity_lookup_table_agent_1 = nn.Embedding(
            self.entity_vocab_size, self.embedding_size)
        if not self.train_relations:
            self.relation_lookup_table_agent_1.weight.requires_grad = False
        if not self.train_entities:
            self.entity_lookup_table_agent_1.weight.requires_grad = False

        # Agent 2 embeddings
        self.relation_lookup_table_agent_2 = nn.Embedding(
            self.action_vocab_size, self.embedding_size)
        self.entity_lookup_table_agent_2 = nn.Embedding(
            self.entity_vocab_size, self.embedding_size)
        if not self.train_relations:
            self.relation_lookup_table_agent_2.weight.requires_grad = False
        if not self.train_entities:
            self.entity_lookup_table_agent_2.weight.requires_grad = False

        # Initialize embeddings
        nn.init.xavier_uniform_(self.relation_lookup_table_agent_1.weight)
        nn.init.xavier_uniform_(self.relation_lookup_table_agent_2.weight)
        self.entity_initializer(self.entity_lookup_table_agent_1.weight)
        self.entity_initializer(self.entity_lookup_table_agent_2.weight)

    def define_agents_policy(self):
        '''
        Defines the agents' policy using LSTM cells.
        '''
        # Agent 1 policy
        self.policy_agent_1 = nn.ModuleList([
            nn.LSTMCell(input_size=self.m * self.embedding_size,
                        hidden_size=self.m * self.embedding_size)
            for _ in range(self.hidden_layers)
        ])

        # Agent 2 policy
        self.policy_agent_2 = nn.ModuleList([
            nn.LSTMCell(input_size=self.m * self.embedding_size,
                        hidden_size=self.m * self.embedding_size)
            for _ in range(self.hidden_layers)
        ])

    def format_state(self, state):
        '''
        Formats the cell- and hidden-state of the LSTM.
        :param state: Tensor [hidden_layers_agent, 2, Batch_size, embedding_size * m]
        :return: List of tuples containing (h, c) states for each layer
        '''
        formatted_state = []
        for i in range(self.hidden_layers):
            h = state[i][0]
            c = state[i][1]
            formatted_state.append((h, c))
        return formatted_state

    def get_mem_shape(self):
        '''
        Returns the shape of the agent's LSTMCell.
        '''
        return (self.hidden_layers, 2, None, self.m * self.embedding_size)

    def get_init_state_array(self, temp_batch_size):
        '''
        Returns initial state arrays for both agents' LSTMCells.
        '''
        mem_agent = self.get_mem_shape()
        agent_mem_1 = np.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3])).astype('float32')
        agent_mem_2 = np.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3])).astype('float32')
        return torch.FloatTensor(agent_mem_1).to(self.device), torch.FloatTensor(agent_mem_2).to(self.device)

    def policy(self, input_action, which_agent):
        '''
        Processes input through the appropriate agent's LSTM policy.
        '''
        if which_agent == 0:
            lstm_cells = self.policy_agent_1
            current_state = self.state_agent_1
        else:
            lstm_cells = self.policy_agent_2
            current_state = self.state_agent_2

        # Process through LSTM layers
        h, c = current_state[0]
        next_states = []
        current_input = input_action

        for i, lstm in enumerate(lstm_cells):
            h, c = lstm(current_input, (h, c))
            next_states.append((h, c))
            current_input = h

        # Update states
        if which_agent == 0:
            self.state_agent_1 = next_states
            self.state_agent_2 = self.state_agent_2
        else:
            self.state_agent_1 = self.state_agent_1
            self.state_agent_2 = next_states

        return h

    def action_encoder_agent(self, next_relations, current_entities, which_agent):
        '''
        Encodes available actions for an agent.
        '''
        if which_agent == 0:
            relation_embedding = self.relation_lookup_table_agent_1(
                next_relations)
            entity_embedding = self.entity_lookup_table_agent_1(
                current_entities)
        else:
            relation_embedding = self.relation_lookup_table_agent_2(
                next_relations)
            entity_embedding = self.entity_lookup_table_agent_2(
                current_entities)

        if self.use_entity_embeddings:
            action_embedding = torch.cat(
                [relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''
        Sets query embeddings for both agents.
        '''

        def set_query_embeddings(self, query_subject, query_relation, query_object):
            # Ensure inputs are PyTorch tensors
            query_subject = torch.tensor(
                query_subject, dtype=torch.long, device=self.device)
            query_relation = torch.tensor(
                query_relation, dtype=torch.long, device=self.device)
            query_object = torch.tensor(
                query_object, dtype=torch.long, device=self.device)

            # Set embeddings for both agents
            self.query_subject_embedding_agent_1 = self.entity_lookup_table_agent_1(
                query_subject)
            self.query_relation_embedding_agent_1 = self.relation_lookup_table_agent_1(
                query_relation)
            self.query_object_embedding_agent_1 = self.entity_lookup_table_agent_1(
                query_object)

            self.query_subject_embedding_agent_2 = self.entity_lookup_table_agent_2(
                query_subject)
            self.query_relation_embedding_agent_2 = self.relation_lookup_table_agent_2(
                query_relation)
            self.query_object_embedding_agent_2 = self.entity_lookup_table_agent_2(
                query_object)

    def step(self, next_relations, next_entities, prev_state_agent_1,
             prev_state_agent_2, prev_relation, current_entities, range_arr, which_agent, random_flag):
        '''
        Computes a step for an agent during the debate.
        '''
        current_entities = torch.tensor(current_entities, dtype=torch.long, device=self.device) \
            if not isinstance(current_entities, torch.Tensor) else current_entities
        prev_relation = torch.tensor(prev_relation, dtype=torch.long, device=self.device) \
            if not isinstance(prev_relation, torch.Tensor) else prev_relation
        next_relations = torch.tensor(next_relations, dtype=torch.long, device=self.device) \
            if not isinstance(next_relations, torch.Tensor) else next_relations
        next_entities = torch.tensor(next_entities, dtype=torch.long, device=self.device) \
            if not isinstance(next_entities, torch.Tensor) else next_entities

        self.state_agent_1 = prev_state_agent_1
        self.state_agent_2 = prev_state_agent_2

        # Get state vector
        if which_agent == 0:
            prev_entity = self.entity_lookup_table_agent_1(current_entities)
            prev_relation_emb = self.relation_lookup_table_agent_1(
                prev_relation)
        else:
            prev_entity = self.entity_lookup_table_agent_2(current_entities)
            prev_relation_emb = self.relation_lookup_table_agent_2(
                prev_relation)

        if self.use_entity_embeddings:
            state = torch.cat([prev_relation_emb, prev_entity], dim=-1)
        else:
            state = prev_relation_emb

        # Get query embeddings based on agent
        if which_agent == 0:
            query_subject_embedding = self.query_subject_embedding_agent_1
            query_relation_embedding = self.query_relation_embedding_agent_1
            query_object_embedding = self.query_object_embedding_agent_1
        else:
            query_subject_embedding = self.query_subject_embedding_agent_2
            query_relation_embedding = self.query_relation_embedding_agent_2
            query_object_embedding = self.query_object_embedding_agent_2

        state_query_concat = torch.cat([
            state, query_subject_embedding, query_relation_embedding, query_object_embedding
        ], dim=-1)

        # Get action embeddings and scores
        candidate_action_embeddings = self.action_encoder_agent(
            next_relations, next_entities, which_agent)
        output = self.policy(state_query_concat, which_agent)
        output_expanded = output.unsqueeze(1)
        prelim_scores = torch.sum(
            candidate_action_embeddings * output_expanded, dim=2)

        # Mask PAD actions
        mask = (next_relations == self.rPAD)
        scores = torch.where(mask, torch.ones_like(
            prelim_scores) * -99999.0, prelim_scores)
        uni_scores = torch.where(mask, torch.ones_like(
            prelim_scores) * -99999.0, torch.ones_like(prelim_scores))

        # Sample action
        if random_flag:
            action = torch.multinomial(torch.softmax(uni_scores, dim=-1), 1)
        else:
            action = torch.multinomial(torch.softmax(scores, dim=-1), 1)

        action_idx = action.squeeze()
        loss = nn.functional.cross_entropy(
            scores, action_idx, reduction='none')

        # Get chosen relation
        batch_indices = torch.arange(
            next_relations.size(0), device=self.device)
        chosen_relation = next_relations[batch_indices, action_idx]

        return (
            loss,
            self.state_agent_1,
            self.state_agent_2,
            nn.functional.log_softmax(scores, dim=-1),
            action_idx,
            chosen_relation
        )

    def forward(self, which_agent, candidate_relation_sequence, candidate_entity_sequence, current_entities, range_arr, T=3, random_flag=None):
        def get_prev_state_agents():
            prev_state_agent_1 = (torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size),
                                  torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size))
            prev_state_agent_2 = (torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size),
                                  torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size))
            return prev_state_agent_1, prev_state_agent_2

        prev_relation = self.dummy_start_label
        argument = self.judge.action_encoder_judge(
            prev_relation, prev_relation)

        all_loss = []
        all_logits = []
        action_idx = []
        all_temp_logits_judge = []
        arguments_representations = []
        all_rewards_agents = []
        all_rewards_before_baseline = []
        prev_state_agent_1, prev_state_agent_2 = get_prev_state_agents()

        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]

            which_agent_t = which_agent[t]
            loss, prev_state_agent_1, prev_state_agent_2, logits, idx, chosen_relation = \
                self.step(next_possible_relations, next_possible_entities,
                          prev_state_agent_1, prev_state_agent_2, prev_relation,
                          current_entities_t, range_arr=range_arr,
                          which_agent=which_agent_t, random_flag=random_flag)

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

            argument = self.judge.extend_argument(argument, torch.tensor(t, dtype=torch.int32), idx, candidate_relation_sequence[t], candidate_entity_sequence[t],
                                                  range_arr)

            if t % self.path_length != (self.path_length - 1):
                all_temp_logits_judge.append(torch.zeros(self.batch_size, 1))
                temp_rewards = torch.zeros(self.batch_size, 1)
                all_rewards_before_baseline.append(temp_rewards)
                all_rewards_agents.append(temp_rewards)
            else:
                logits_judge, rep_argu = self.judge.classify_argument(argument)
                rewards = torch.sigmoid(logits_judge)
                all_temp_logits_judge.append(logits_judge)
                arguments_representations.append(rep_argu)
                all_rewards_before_baseline.append(rewards)
                if self.custom_baseline:
                    no_op_arg = self.judge.action_encoder_judge(
                        prev_relation, prev_relation)
                    for i in range(self.path_length):
                        no_op_arg = self.judge.extend_argument(no_op_arg, torch.tensor(i, dtype=torch.int32), torch.zeros_like(idx),
                                                               candidate_relation_sequence[0], candidate_entity_sequence[0],
                                                               range_arr)
                    no_op_logits, rep_argu = self.judge.classify_argument(
                        no_op_arg)
                    rewards_no_op = torch.sigmoid(no_op_logits)
                    all_rewards_agents.append(rewards - rewards_no_op)
                else:
                    all_rewards_agents.append(rewards)

        loss_judge, final_logit_judge = self.judge.final_loss(
            arguments_representations)
        return loss_judge, final_logit_judge, all_temp_logits_judge, all_loss, all_logits, action_idx, all_rewards_agents, all_rewards_before_baseline

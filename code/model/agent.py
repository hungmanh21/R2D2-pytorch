import numpy as np
import torch
import torch.nn as nn


class Agent(nn.Module):
    '''
    Class for the agents in R2D2. Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.
    A single instance of Agent contains both the pro and con agent.
    '''

    def __init__(self, params, judge):
        '''
        Initializes the agents.
        :param params: Dict. Parameters of the experiment.
        :param judge: Judge. Instance of Judge that the agents present arguments to.
        '''
        super(Agent, self).__init__()

        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.test_rollouts = params['test_rollouts']
        self.path_length = params['path_length']
        self.batch_size = params['batch_size'] * \
            (1 + params['false_facts_train']) * params['num_rollouts']
        self.dummy_start_label = torch.ones(
            self.batch_size, dtype=torch.long) * params['relation_vocab']['DUMMY_START_RELATION']

        self.hidden_layers = params['layers_agent']
        self.custom_baseline = params['custom_baseline']
        self.use_entity_embeddings = params['use_entity_embeddings']

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.use_entity_embeddings:
            self.m = 2
            self.entity_init = nn.init.xavier_uniform_
        else:
            self.m = 1
            self.entity_init = nn.init.zeros_

        self.define_embeddings()
        self.define_agents_policy()

    def define_embeddings(self):
        '''
        Creates embeddings for both agents for the KG's relations and entities.
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
        self.entity_init(self.entity_lookup_table_agent_1.weight)
        self.entity_init(self.entity_lookup_table_agent_2.weight)

    def define_agents_policy(self):
        '''
        Defines the agents' policy using LSTM cells.
        '''
        # Agent 1 policy
        self.policy_agent_1 = nn.ModuleList([
            nn.LSTM(self.m * self.embedding_size,
                    self.m * self.embedding_size, 1)
            for _ in range(self.hidden_layers)
        ])

        # Agent 2 policy
        self.policy_agent_2 = nn.ModuleList([
            nn.LSTM(self.m * self.embedding_size,
                    self.m * self.embedding_size, 1)
            for _ in range(self.hidden_layers)
        ])

    def init_hidden(self, batch_size, device):
        '''
        Initializes hidden states for both agents' LSTMs.
        '''
        hidden_size = self.m * self.embedding_size

        # Initialize hidden states for both agents
        h1 = [torch.zeros(1, batch_size, hidden_size).to(device)
              for _ in range(self.hidden_layers)]
        c1 = [torch.zeros(1, batch_size, hidden_size).to(device)
              for _ in range(self.hidden_layers)]
        h2 = [torch.zeros(1, batch_size, hidden_size).to(device)
              for _ in range(self.hidden_layers)]
        c2 = [torch.zeros(1, batch_size, hidden_size).to(device)
              for _ in range(self.hidden_layers)]

        return (h1, c1), (h2, c2)

    def action_encoder_agent(self, next_relations, current_entities, which_agent):
        '''
        Encodes available actions for an agent.

        :param next_relations: Tensor [Batch_size, max_num_actions]
        :param current_entities: Tensor [Batch_size, max_num_actions]
        :param which_agent: Tensor []
        :return: Tensor [Batch_size, max_num_actions, embedding_size * m]
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
            return torch.cat([relation_embedding, entity_embedding], dim=-1)
        return relation_embedding

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''
        Sets query embeddings for both agents.
        '''
        print("In set query")
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

    def policy(self, input_action, which_agent, hidden_states):
        '''
        Processes input through the appropriate agent's LSTM.

        :param input_action: Tensor [Batch_size, input_size]
        :param which_agent: int (0 or 1)
        :param hidden_states: Tuple of hidden states for both agents
        :return: LSTM output and updated hidden states
        '''
        state_agent_1, state_agent_2 = hidden_states

        if which_agent == 0:
            policy = self.policy_agent_1
            current_state = state_agent_1
        else:
            policy = self.policy_agent_2
            current_state = state_agent_2

        x = input_action.unsqueeze(0)  # Add sequence dimension

        # Process through LSTM layers
        new_h, new_c = [], []
        for i, lstm in enumerate(policy):
            x, (h, c) = lstm(x, (current_state[0][i], current_state[1][i]))
            new_h.append(h)
            new_c.append(c)

        # Update appropriate agent's state
        if which_agent == 0:
            state_agent_1 = (new_h, new_c)
        else:
            state_agent_2 = (new_h, new_c)

        return x.squeeze(0), (state_agent_1, state_agent_2)

    def step(self, next_relations, next_entities, hidden_states, prev_relation,
             current_entities, which_agent, random_flag):
        '''
        Computes a single step for an agent during debate.
        '''
        device = next_relations.device
        batch_size = next_relations.size(0)

        # Get embeddings for current state
        if which_agent == 0:
            prev_entity = self.entity_lookup_table_agent_1(current_entities)
            prev_relation_emb = self.relation_lookup_table_agent_1(
                prev_relation)
            query_subject = self.query_subject_embedding_agent_1
            query_relation = self.query_relation_embedding_agent_1
            query_object = self.query_object_embedding_agent_1
        else:
            prev_entity = self.entity_lookup_table_agent_2(current_entities)
            prev_relation_emb = self.relation_lookup_table_agent_2(
                prev_relation)
            query_subject = self.query_subject_embedding_agent_2
            query_relation = self.query_relation_embedding_agent_2
            query_object = self.query_object_embedding_agent_2

        # Prepare state input
        if self.use_entity_embeddings:
            state = torch.cat([prev_relation_emb, prev_entity], dim=-1)
        else:
            state = prev_relation_emb

        state = torch.cat(
            [state, query_subject, query_relation, query_object], dim=-1)

        # Get policy output and action embeddings
        candidate_actions = self.action_encoder_agent(
            next_relations, next_entities, which_agent)
        output, new_hidden_states = self.policy(
            state, which_agent, hidden_states)

        # Calculate action scores
        scores = torch.sum(candidate_actions * output.unsqueeze(1), dim=-1)

        # Mask PAD actions
        mask = (next_relations == self.rPAD)
        scores = scores.masked_fill(mask, float('-inf'))

        # Sample action
        if random_flag:
            # Random sampling
            valid_actions = ~mask
            probs = torch.zeros_like(scores).masked_fill(valid_actions, 1.0)
            dist = torch.distributions.Categorical(probs)
        else:
            # Policy-based sampling
            dist = torch.distributions.Categorical(logits=scores)

        action = dist.sample()

        # Calculate loss
        loss = -dist.log_prob(action)

        # Get chosen relation
        batch_indices = torch.arange(batch_size, device=device)
        chosen_relation = next_relations[batch_indices, action]

        return (loss, new_hidden_states, torch.log_softmax(scores, dim=-1),
                action, chosen_relation)

    def forward(self, which_agent, candidate_relation_sequence, candidate_entity_sequence,
                current_entities, T=3, random_flag=False):
        '''
        Runs a complete debate sequence.
        '''
        device = candidate_relation_sequence[0].device
        batch_size = candidate_relation_sequence[0].size(0)

        # Initialize states and tracking lists
        hidden_states = self.init_hidden(batch_size, device)
        prev_relation = self.dummy_start_label.to(device)
        argument = self.judge.action_encoder_judge(
            prev_relation, prev_relation)

        all_loss = []
        all_logits = []
        action_idx = []
        all_temp_logits_judge = []
        arguments_representations = []
        all_rewards_agents = []
        all_rewards_before_baseline = []

        # Run through debate steps
        for t in range(T):
            next_relations = candidate_relation_sequence[t]
            next_entities = candidate_entity_sequence[t]
            current_ents = current_entities[t]
            which_agent_t = which_agent[t]

            # Get step results
            loss, hidden_states, logits, idx, chosen_relation = self.step(
                next_relations, next_entities, hidden_states, prev_relation,
                current_ents, which_agent_t, random_flag
            )

            # Update tracking
            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

            # Update argument
            batch_indices = torch.arange(batch_size, device=device)
            argument = self.judge.extend_argument(
                argument, t, idx, candidate_relation_sequence[t],
                candidate_entity_sequence[t], batch_indices
            )

            # Handle rewards and judge logits
            if t % self.path_length != (self.path_length - 1):
                # Not end of argument
                all_temp_logits_judge.append(
                    torch.zeros((batch_size, 1), device=device))
                temp_rewards = torch.zeros((batch_size, 1), device=device)
                all_rewards_before_baseline.append(temp_rewards)
                all_rewards_agents.append(temp_rewards)
            else:
                # End of argument
                logits_judge, rep_argu = self.judge.classify_argument(argument)
                rewards = torch.sigmoid(logits_judge)
                all_temp_logits_judge.append(logits_judge)
                arguments_representations.append(rep_argu)
                all_rewards_before_baseline.append(rewards)

                if self.custom_baseline:
                    # Calculate baseline rewards using no-op argument
                    no_op_arg = self.judge.action_encoder_judge(
                        prev_relation, prev_relation)
                    for i in range(self.path_length):
                        no_op_arg = self.judge.extend_argument(
                            no_op_arg, i, torch.zeros_like(idx),
                            candidate_relation_sequence[0],
                            candidate_entity_sequence[0], batch_indices
                        )
                    no_op_logits, _ = self.judge.classify_argument(no_op_arg)
                    rewards_no_op = torch.sigmoid(no_op_logits)
                    all_rewards_agents.append(rewards - rewards_no_op)
                else:
                    all_rewards_agents.append(rewards)

        # Calculate final judge loss and logits
        loss_judge, final_logit_judge = self.judge.final_loss(
            arguments_representations)

        return (loss_judge, final_logit_judge, all_temp_logits_judge, all_loss,
                all_logits, action_idx, all_rewards_agents, all_rewards_before_baseline)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Agent(nn.Module):
    def __init__(self, params, judge):
        super(Agent, self).__init__()

        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']

        self.ePAD = torch.tensor(
            params['entity_vocab']['PAD'], dtype=torch.long)
        self.rPAD = torch.tensor(
            params['relation_vocab']['PAD'], dtype=torch.long)

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

        if self.use_entity_embeddings:
            self.m = 2
            self.entity_init_method = nn.init.xavier_uniform_
        else:
            self.m = 1
            self.entity_init_method = torch.zeros_like

        self.define_embeddings(params)
        self.define_agents_policy()

    def define_embeddings(self, params):
        # Relation embeddings for both agents
        self.relation_embedding_1 = nn.Embedding(
            self.action_vocab_size,
            self.embedding_size,
            padding_idx=self.rPAD
        )
        nn.init.xavier_uniform_(self.relation_embedding_1.weight)
        self.relation_embedding_1.weight.requires_grad = self.train_relations

        self.relation_embedding_2 = nn.Embedding(
            self.action_vocab_size,
            self.embedding_size,
            padding_idx=self.rPAD
        )
        nn.init.xavier_uniform_(self.relation_embedding_2.weight)
        self.relation_embedding_2.weight.requires_grad = self.train_relations

        # Entity embeddings for both agents
        self.entity_embedding_1 = nn.Embedding(
            self.entity_vocab_size,
            self.embedding_size,
            padding_idx=self.ePAD
        )
        self.entity_init_method(self.entity_embedding_1.weight)
        self.entity_embedding_1.weight.requires_grad = self.train_entities

        self.entity_embedding_2 = nn.Embedding(
            self.entity_vocab_size,
            self.embedding_size,
            padding_idx=self.ePAD
        )
        self.entity_init_method(self.entity_embedding_2.weight)
        self.entity_embedding_2.weight.requires_grad = self.train_entities

    def define_agents_policy(self):
        # Create LSTM cells for each agent
        lstm_layers = []
        for _ in range(self.hidden_layers):
            lstm_layers.append(nn.LSTMCell(
                input_size=self.m * self.embedding_size,
                hidden_size=self.m * self.embedding_size
            ))

        self.policy_agent_1_cells = nn.ModuleList(lstm_layers)

        lstm_layers = []
        for _ in range(self.hidden_layers):
            lstm_layers.append(nn.LSTMCell(
                input_size=self.m * self.embedding_size,
                hidden_size=self.m * self.embedding_size
            ))

        self.policy_agent_2_cells = nn.ModuleList(lstm_layers)

    def get_mem_shape(self):
        return (self.hidden_layers, 2, None, self.m * self.embedding_size)

    def get_init_state_array(self, temp_batch_size):
        mem_agent = self.get_mem_shape()
        agent_mem_1 = torch.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size *
             self.test_rollouts, mem_agent[3])
        ).float()
        agent_mem_2 = torch.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size *
             self.test_rollouts, mem_agent[3])
        ).float()
        return agent_mem_1, agent_mem_2

    def action_encoder_agent(self, next_relations, current_entities, which_agent):
        if which_agent == 0:
            relation_embedding = self.relation_embedding_1(next_relations)
            entity_embedding = self.entity_embedding_1(current_entities)
        else:
            relation_embedding = self.relation_embedding_2(next_relations)
            entity_embedding = self.entity_embedding_2(current_entities)

        if self.use_entity_embeddings:
            action_embedding = torch.cat(
                [relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        self.query_subject_embedding_1 = self.entity_embedding_1(query_subject)
        self.query_relation_embedding_1 = self.relation_embedding_1(
            query_relation)
        self.query_object_embedding_1 = self.entity_embedding_1(query_object)

        self.query_subject_embedding_2 = self.entity_embedding_2(query_subject)
        self.query_relation_embedding_2 = self.relation_embedding_2(
            query_relation)
        self.query_object_embedding_2 = self.entity_embedding_2(query_object)

    def policy(self, input_action, which_agent, prev_states):
        """
        Process input through the appropriate agent's policy (LSTM cells)

        :param input_action: Input action embedding
        :param which_agent: 0 for agent 1, 1 for agent 2
        :param prev_states: Previous LSTM cell states
        :return: New output and updated states
        """
        if which_agent == 0:
            cells = self.policy_agent_1_cells
        else:
            cells = self.policy_agent_2_cells

        # Process through each LSTM cell layer
        new_states = []
        output = input_action

        for i, cell in enumerate(cells):
            h, c = prev_states[i]
            h, c = cell(output, (h, c))
            new_states.append((h, c))
            output = h

        return output, new_states

    def step(self, next_relations, next_entities, prev_state_agent_1,
             prev_state_agent_2, prev_relation, current_entities,
             range_arr, which_agent, random_flag):
        """
        Compute a single step for an agent during the debate
        """
        # Determine which agent's embeddings to use
        if which_agent == 0:
            prev_entity_emb = self.entity_embedding_1(current_entities)
            prev_relation_emb = self.relation_embedding_1(prev_relation)
            query_subject = self.query_subject_embedding_1
            query_relation = self.query_relation_embedding_1
            query_object = self.query_object_embedding_1
            states = prev_state_agent_1
            cells = self.policy_agent_1_cells
        else:
            prev_entity_emb = self.entity_embedding_2(current_entities)
            prev_relation_emb = self.relation_embedding_2(prev_relation)
            query_subject = self.query_subject_embedding_2
            query_relation = self.query_relation_embedding_2
            query_object = self.query_object_embedding_2
            states = prev_state_agent_2
            cells = self.policy_agent_2_cells

        # Prepare state input
        if self.use_entity_embeddings:
            state = torch.cat([prev_relation_emb, prev_entity_emb], dim=-1)
        else:
            state = prev_relation_emb

        # Concatenate state with query information
        state_query_concat = torch.cat([
            state,
            query_subject,
            query_relation,
            query_object
        ], dim=-1)

        # Get candidate action embeddings
        candidate_action_embeddings = self.action_encoder_agent(
            next_relations, current_entities, which_agent
        )

        # Process through policy (LSTM)
        output, new_states = self.policy(
            state_query_concat, which_agent, states)

        # Compute action scores
        output_expanded = output.unsqueeze(1)  # [B, 1, 2D]
        prelim_scores = torch.sum(
            candidate_action_embeddings * output_expanded, dim=-1)

        # Mask PAD actions
        mask = (next_relations == self.rPAD)
        scores = prelim_scores.masked_fill(mask, float('-inf'))
        uni_scores = prelim_scores.masked_fill(mask, float('-inf'))
        uni_scores = torch.ones_like(scores)

        # Sample action
        if random_flag:
            action = torch.multinomial(F.softmax(uni_scores, dim=-1), 1)
        else:
            action = torch.multinomial(F.softmax(scores, dim=-1), 1)

        # Compute loss
        label_action = action.squeeze(1)
        loss = F.cross_entropy(scores, label_action)

        # Get chosen relation
        batch_indices = torch.arange(range_arr.size(0))
        chosen_relation = next_relations[batch_indices, action.squeeze()]

        return (
            loss,
            new_states if which_agent == 0 else prev_state_agent_1,
            new_states if which_agent == 1 else prev_state_agent_2,
            F.log_softmax(scores, dim=-1),
            action.squeeze(),
            chosen_relation
        )

    def forward(self, which_agent, candidate_relation_sequence, candidate_entity_sequence,
                current_entities, range_arr, T=3, random_flag=None):
        """
        Construct a whole debate
        """
        # Initialize states
        prev_state_agent_1 = [
            (torch.zeros(self.batch_size, self.m * self.embedding_size).to(which_agent.device),
             torch.zeros(self.batch_size, self.m * self.embedding_size).to(which_agent.device))
            for _ in range(self.hidden_layers)
        ]
        prev_state_agent_2 = [
            (torch.zeros(self.batch_size, self.m * self.embedding_size).to(which_agent.device),
             torch.zeros(self.batch_size, self.m * self.embedding_size).to(which_agent.device))
            for _ in range(self.hidden_layers)
        ]

        prev_relation = self.dummy_start_label
        argument = self.judge.action_encoder_judge(
            prev_relation, prev_relation)

        # Initialize lists to store debate progression
        all_loss = []
        all_logits = []
        action_idx = []
        all_temp_logits_judge = []
        arguments_representations = []
        all_rewards_agents = []
        all_rewards_before_baseline = []

        # Debate progression
        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]
            which_agent_t = which_agent[t]

            # Perform step
            loss, prev_state_agent_1, prev_state_agent_2, logits, idx, chosen_relation = self.step(
                next_possible_relations, next_possible_entities,
                prev_state_agent_1, prev_state_agent_2,
                prev_relation, current_entities_t,
                range_arr=range_arr,
                which_agent=which_agent_t,
                random_flag=random_flag
            )

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

            # Extend argument
            argument = self.judge.extend_argument(
                argument,
                torch.tensor(t, dtype=torch.int32),
                idx,
                candidate_relation_sequence[t],
                candidate_entity_sequence[t],
                range_arr
            )

            # Handle rewards and judge logits
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
                        no_op_arg = self.judge.extend_argument(
                            no_op_arg,
                            torch.tensor(i, dtype=torch.int32),
                            torch.zeros_like(idx),
                            candidate_relation_sequence[0],
                            candidate_entity_sequence[0],
                            range_arr
                        )
                    no_op_logits, rep_argu = self.judge.classify_argument(
                        no_op_arg)
                    rewards_no_op = torch.sigmoid(no_op_logits)
                    all_rewards_agents.append(rewards - rewards_no_op)
                else:
                    all_rewards_agents.append(rewards)

        # Final judge loss
        loss_judge, final_logit_judge = self.judge.final_loss(
            arguments_representations)

        return (
            loss_judge,
            final_logit_judge,
            all_temp_logits_judge,
            all_loss,
            all_logits,
            action_idx,
            all_rewards_agents,
            all_rewards_before_baseline
        )

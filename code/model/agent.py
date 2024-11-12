import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    """
    A single instance of Agent contains both the pro and con agent.
    """
    def __init__(self, params, judge):
        super(Agent, self).__init__()
        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.ePAD = torch.tensor(params['entity_vocab']['PAD'], dtype=torch.int32)
        self.rPAD = torch.tensor(params['relation_vocab']['PAD'], dtype=torch.int32)
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.test_rollouts = params['test_rollouts']
        self.path_length = params['path_length']
        self.batch_size = params['batch_size'] * (1 + params['false_facts_train']) * params['num_rollouts']
        self.dummy_start_label = torch.ones(self.batch_size, dtype=torch.int64) * params['relation_vocab']['DUMMY_START_RELATION']

        self.hidden_layers = params['layers_agent']
        self.custom_baseline = params['custom_baseline']
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.entity_initializer = nn.init.xavier_uniform_
            self.m = 2
        else:
            self.m = 1
            self.entity_initializer = nn.init.zeros_

        self.define_embeddings()
        self.define_agents_policy()

    def define_embeddings(self):
        self.relation_lookup_table_agent_1 = nn.Embedding(
            self.action_vocab_size, self.embedding_size
        )
        self.relation_lookup_table_agent_2 = nn.Embedding(
            self.action_vocab_size, self.embedding_size
        )
        self.entity_lookup_table_agent_1 = nn.Embedding(
            self.entity_vocab_size, self.embedding_size
        )
        self.entity_lookup_table_agent_2 = nn.Embedding(
            self.entity_vocab_size, self.embedding_size
        )

        nn.init.xavier_uniform_(self.relation_lookup_table_agent_1.weight)
        nn.init.xavier_uniform_(self.relation_lookup_table_agent_2.weight)

        self.entity_initializer(self.entity_lookup_table_agent_1.weight)
        self.entity_initializer(self.entity_lookup_table_agent_2.weight)

    def define_agents_policy(self):
        self.policy_agent_1 = nn.LSTM(self.m * self.embedding_size, self.m * self.embedding_size, self.hidden_layers, batch_first=True)
        self.policy_agent_2 = nn.LSTM(self.m * self.embedding_size, self.m * self.embedding_size, self.hidden_layers, batch_first=True)

    def format_state(self, state):
        return [state[:, i] for i in range(state.size(1))]

    def get_mem_shape(self):
        return (self.hidden_layers, 2, None, self.m * self.embedding_size)

    def get_init_state_array(self, temp_batch_size):
        mem_agent = self.get_mem_shape()
        agent_mem_1 = torch.zeros(mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3]).type(torch.FloatTensor)
        agent_mem_2 = torch.zeros(mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3]).type(torch.FloatTensor)
        return agent_mem_1, agent_mem_2

    def policy(self, input_action, which_agent):
        def policy_1():
            return self.policy_agent_1(input_action, self.state_agent_1)

        def policy_2():
            return self.policy_agent_2(input_action, self.state_agent_2)

        output, new_state = torch.where(which_agent == 0, policy_1(), policy_2())
        new_state_stacked = torch.stack(new_state)
        state_agent_1_stacked = torch.stack(self.state_agent_1)
        state_agent_1_stacked = (1-which_agent)*new_state_stacked + which_agent* state_agent_1_stacked
        self.state_agent_1 = self.format_state(state_agent_1_stacked)

        state_agent_2_stacked = torch.stack(self.state_agent_2)
        state_agent_2_stacked = which_agent*new_state_stacked + (1-which_agent)* state_agent_2_stacked
        self.state_agent_2 = self.format_state(state_agent_2_stacked)

        return output

    def action_encoder_agent(self, next_relations, current_entities, which_agent):
        relation_embedding = torch.where(which_agent == 0, self.relation_lookup_table_agent_1, self.relation_lookup_table_agent_2)
        entity_embedding = torch.where(which_agent == 0, self.entity_lookup_table_agent_1, self.entity_lookup_table_agent_2)

        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding[next_relations], entity_embedding[current_entities]], dim=-1)
        else:
            action_embedding = relation_embedding[next_relations]

        return action_embedding

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        self.query_subject_embedding_agent_1 = self.entity_lookup_table_agent_1[query_subject]
        self.query_relation_embedding_agent_1 = self.relation_lookup_table_agent_1[query_relation]
        self.query_object_embedding_agent_1 = self.entity_lookup_table_agent_1[query_object]

        self.query_subject_embedding_agent_2 = self.entity_lookup_table_agent_2[query_subject]
        self.query_relation_embedding_agent_2 = self.relation_lookup_table_agent_2[query_relation]
        self.query_object_embedding_agent_2 = self.entity_lookup_table_agent_2[query_object]

    def step(self, next_relations, next_entities, prev_state_agent_1, prev_state_agent_2, prev_relation, current_entities, range_arr, which_agent, random_flag):
        self.state_agent_1 = prev_state_agent_1
        self.state_agent_2 = prev_state_agent_2
        is_agent_1 = which_agent == 0

        prev_entity = torch.where(is_agent_1, self.entity_lookup_table_agent_1[current_entities], self.entity_lookup_table_agent_2[current_entities])
        prev_relation = torch.where(is_agent_1, self.relation_lookup_table_agent_1[prev_relation], self.relation_lookup_table_agent_2[prev_relation])

        if self.use_entity_embeddings:
            state = torch.cat([prev_relation, prev_entity], dim=-1)
        else:
            state = prev_relation

        def get_policy_state():
            query_subject_embedding = self.query_subject_embedding_agent_1 if is_agent_1 else self.query_subject_embedding_agent_2
            query_relation_embedding = self.query_relation_embedding_agent_1 if is_agent_1 else self.query_relation_embedding_agent_2
            query_object_embedding = self.query_object_embedding_agent_1 if is_agent_1 else self.query_object_embedding_agent_2

            state_query_concat = torch.cat([state, query_subject_embedding, query_relation_embedding, query_object_embedding], dim=-1)
            return state_query_concat

        #TODO : check lai code
        candidate_action_embeddings = self.action_encoder_agent(next_relations, current_entities, which_agent)
        output, (new_hidden, new_cell) = self.policy(get_policy_state(), which_agent)
        # self.state_agent_1 = (new_hidden, new_cell)
        # self.state_agent_2 = (new_hidden, new_cell)

        output_expanded = output.unsqueeze(1)
        prelim_scores = torch.sum(output_expanded * candidate_action_embeddings, dim=2)

        comparison_tensor = torch.ones_like(next_relations, dtype=torch.int32) * self.rPAD
        mask = next_relations == comparison_tensor
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0
        scores = torch.where(mask, dummy_scores, prelim_scores)
        uni_scores = torch.where(mask, dummy_scores, torch.ones_like(prelim_scores))

        action = torch.multinomial(F.softmax(scores, dim=1), 1).squeeze(1)
        action = torch.where(random_flag, torch.multinomial(F.softmax(uni_scores, dim=1), 1).squeeze(1), action)

        label_action = action
        loss = F.cross_entropy(scores, label_action)

        action_idx = action
        chosen_relation = next_relations[range_arr, action_idx]

        return loss, self.state_agent_1, self.state_agent_2, F.log_softmax(scores, dim=1), action_idx, chosen_relation

    def forward(self, which_agent, candidate_relation_sequence, candidate_entity_sequence, current_entities, range_arr, T=3, random_flag=None):
        def get_prev_state_agents():
            prev_state_agent_1 = (torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size),
                                  torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size))
            prev_state_agent_2 = (torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size),
                                  torch.zeros(self.hidden_layers, self.batch_size, self.m * self.embedding_size))
            return prev_state_agent_1, prev_state_agent_2

        prev_relation = self.dummy_start_label
        argument = self.judge.action_encoder_judge(prev_relation, prev_relation)

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
                    no_op_arg = self.judge.action_encoder_judge(prev_relation, prev_relation)
                    for i in range(self.path_length):
                        no_op_arg = self.judge.extend_argument(no_op_arg, torch.tensor(i, dtype=torch.int32), torch.zeros_like(idx),
                                                               candidate_relation_sequence[0], candidate_entity_sequence[0],
                                                               range_arr)
                    no_op_logits, rep_argu = self.judge.classify_argument(no_op_arg)
                    rewards_no_op = torch.sigmoid(no_op_logits)
                    all_rewards_agents.append(rewards - rewards_no_op)
                else:
                    all_rewards_agents.append(rewards)

        loss_judge, final_logit_judge = self.judge.final_loss(arguments_representations)
        return loss_judge, final_logit_judge, all_temp_logits_judge, all_loss, all_logits, action_idx, all_rewards_agents, all_rewards_before_baseline
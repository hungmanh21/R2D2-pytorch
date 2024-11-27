import datetime
from pprint import pprint
import sys
import uuid
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import json
import numpy as np
from sklearn.model_selection import ParameterGrid
from code.model.agent import Agent
from code.model.environment import Environment
from code.model.judge import Judge
from code.options import read_options
from debate_printer import Debate_Printer

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer:
    def __init__(self, params, best_metric):
        self.params = params
        self.best_metric = best_metric
        self.batch_size = params['batch_size'] * \
            (1 + params['false_facts_train']) * params['num_rollouts']
        self.judge = Judge(params)
        self.agent = Agent(params, self.judge)
        self.train_environment = Environment(params, 'train')
        self.dev_test_environment = Environment(params, 'dev')
        self.test_test_environment = Environment(params, 'test')
        self.number_steps = params['path_length'] * \
            params['number_arguments'] * 2
        self.best_metric = best_metric
        self.learning_rate_judge_init = params['learning_rate_judge']
        self.optimizer_judge = optim.Adam(
            self.judge.parameters(), lr=self.learning_rate_judge_init)
        self.optimizer_agents = optim.Adam(
            list(self.agent.parameters()), lr=params['learning_rate_agents'])

    def set_random_seeds(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def calc_reinforce_loss(self, rewards_1, rewards_2, which_agent_sequence):
        rewards_1 = torch.tensor(rewards_1)
        rewards_2 = torch.tensor(rewards_2)

        # Normalize rewards
        rewards_1 = (rewards_1 - rewards_1.mean()) / (rewards_1.std() + 1e-6)
        rewards_2 = (rewards_2 - rewards_2.mean()) / (rewards_2.std() + 1e-6)

        loss_1 = torch.mean(torch.mul(torch.stack(
            self.per_example_loss[::2]), rewards_1))
        loss_2 = torch.mean(torch.mul(torch.stack(
            self.per_example_loss[1::2]), rewards_2))

        entropy_policy_1, entropy_policy_2 = self.entropy_reg_loss(
            self.per_example_logits)
        total_loss_1 = loss_1 - self.params['decaying_beta'] * entropy_policy_1
        total_loss_2 = loss_2 - self.params['decaying_beta'] * entropy_policy_2

        return total_loss_1, total_loss_2

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        mask = torch.tensor(self.which_agent_sequence, dtype=torch.bool)
        mask = mask.unsqueeze(0).expand(all_logits.size(1), -1)
        mask = mask.unsqueeze(0).expand(all_logits.size(0), -1, -1)
        logits_1 = all_logits[~mask].reshape(
            all_logits.size(0), all_logits.size(1), -1)
        logits_2 = all_logits[mask].reshape(
            all_logits.size(0), all_logits.size(1), -1)

        entropy_policy_1 = - \
            torch.mean(
                torch.sum(torch.mul(torch.exp(logits_1), logits_1), dim=1))
        entropy_policy_2 = - \
            torch.mean(
                torch.sum(torch.mul(torch.exp(logits_2), logits_2), dim=1))

        return entropy_policy_1, entropy_policy_2

    def initialize(self, restore=None):
        self.which_agent_sequence = []
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.input_path = []
        self.entity_sequence = []

        self.query_subject = torch.zeros(self.batch_size, dtype=torch.long)
        self.query_relation = torch.zeros(self.batch_size, dtype=torch.long)
        self.query_object = torch.zeros(self.batch_size, dtype=torch.long)
        self.labels = torch.zeros(self.batch_size, 1, dtype=torch.float32)
        self.judge.set_labels_placeholder(self.labels)

        self.random_flag = torch.tensor(False)
        self.range_arr = torch.arange(self.batch_size)
        self.global_step = 0
        self.decaying_beta = self.params['beta'] * \
            0.9 ** (self.global_step // 200)

        self.judge.set_query_embeddings(
            self.query_subject, self.query_relation, self.query_object)
        self.agent.set_query_embeddings(
            self.query_subject, self.query_relation, self.query_object)

        for t in range(self.number_steps):
            which_agent = torch.tensor(0.0)
            next_possible_relations = torch.zeros(
                self.batch_size, self.params['max_num_actions'], dtype=torch.long)
            next_possible_entities = torch.zeros(
                self.batch_size, self.params['max_num_actions'], dtype=torch.long)
            input_label_relation = torch.zeros(
                self.batch_size, dtype=torch.long)
            start_entities = torch.zeros(self.batch_size, dtype=torch.long)
            self.which_agent_sequence.append(which_agent)
            self.input_path.append(input_label_relation)
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)

        self.loss_judge, self.final_logits_judge, self.temp_logits_judge, self.per_example_loss, self.per_example_logits, \
            self.action_idx, self.rewards_agents, self.rewards_before_baseline = self.agent(
                self.which_agent_sequence, self.candidate_relation_sequence, self.candidate_entity_sequence,
                self.entity_sequence, self.range_arr, self.number_steps, self.random_flag)

        self.train_judge = self.optimizer_judge.step(self.loss_judge)
        self.loss_1, self.loss_2 = self.calc_reinforce_loss(
            self.rewards_agents[::2], self.rewards_agents[1::2], self.which_agent_sequence)
        self.train_op_1 = self.optimizer_agents.step(self.loss_1)
        self.train_op_2 = self.optimizer_agents.step(self.loss_2)

    def train(self):
        self.batch_counter = 0
        debate_printer = Debate_Printer(
            self.params['output_dir'], self.train_environment.grapher, self.params['num_rollouts'])

        for episode in self.train_environment.get_episodes():
            is_train_judge = (self.batch_counter // self.params['train_judge_every']
                              ) % 2 == 0 or self.batch_counter >= self.params['rounds_sub_training']
            self.batch_counter += 1

            episode_answers = episode.get_labels()
            debate_printer.create_debates(episode.get_query_subjects(
            ), episode.get_query_relation(), episode.get_query_objects(), episode_answers)

            self.query_subject.copy_(episode.get_query_subjects())
            self.query_relation.copy_(episode.get_query_relation())
            self.query_object.copy_(episode.get_query_objects())
            self.labels.copy_(episode_answers)

            loss_before_regularization = []
            logits = []
            i = 0
            debate_printer_rel_list = []
            debate_printer_ent_list = []

            for arguments in range(self.params['number_arguments']):
                state = episode.reset_initial_state()
                for path_num in range(self.params['path_length']):
                    self.which_agent_sequence[i] = 0.0
                    self.candidate_relation_sequence[i].copy_(
                        state['next_relations'])
                    self.candidate_entity_sequence[i].copy_(
                        state['next_entities'])
                    self.entity_sequence[i].copy_(state['current_entities'])
                    temp_logits_judge, per_example_loss, per_example_logits, idx, rewards, print_rewards = self.agent(
                        self.candidate_relation_sequence[i], self.candidate_entity_sequence[i], self.agent.get_init_state_array(self.batch_size)[0], self.agent.get_init_state_array(self.batch_size)[1], self.params['prev_relation'], state['current_entities'], self.range_arr, 0.0, self.random_flag)
                    loss_before_regularization.append(per_example_loss)
                    rel_string, ent_string = debate_printer.get_action_rel_ent(
                        idx, state)
                    debate_printer_rel_list.append(rel_string)
                    debate_printer_ent_list.append(ent_string)
                    state = episode(idx)
                    i += 1
                    logits.append((0, rewards))

                debate_printer.create_arguments(
                    debate_printer_rel_list, debate_printer_ent_list, rewards, True)
                debate_printer_rel_list.clear()
                debate_printer_ent_list.clear()

                state = episode.reset_initial_state()
                for path_num in range(self.params['path_length']):
                    self.which_agent_sequence[i] = 1.0
                    self.candidate_relation_sequence[i].copy_(
                        state['next_relations'])
                    self.candidate_entity_sequence[i].copy_(
                        state['next_entities'])
                    self.entity_sequence[i].copy_(state['current_entities'])
                    temp_logits_judge, per_example_loss, per_example_logits, idx, rewards, print_rewards = self.agent(
                        self.candidate_relation_sequence[i], self.candidate_entity_sequence[i], self.agent.get_init_state_array(self.batch_size)[0], self.agent.get_init_state_array(self.batch_size)[1], self.params['prev_relation'], state['current_entities'], self.range_arr, 1.0, self.random_flag)
                    loss_before_regularization.append(per_example_loss)
                    rel_string, ent_string = debate_printer.get_action_rel_ent(
                        idx, state)
                    debate_printer_rel_list.append(rel_string)
                    debate_printer_ent_list.append(ent_string)
                    state = episode(idx)
                    i += 1
                    logits.append((1, rewards))

                debate_printer.create_arguments(
                    debate_printer_rel_list, debate_printer_ent_list, rewards, False)
                debate_printer_rel_list.clear()
                debate_printer_ent_list.clear()

            logits_judge = self.judge(self.final_logits_judge)
            debate_printer.set_debates_final_logit(logits_judge)

            if is_train_judge:
                self.train_judge
                self.learning_rate_judge = self.learning_rate_judge_init
            else:
                print("judge is NOT trained \n")

            predictions = logits_judge > 0
            if self.batch_counter % self.params['save_debate_every'] == 0:
                debate_printer.write(
                    f'argument_train_{self.batch_counter}.txt')

            acc = predictions.float().mean()
            logger.info(f"Mean label === {episode_answers.float().mean()}")
            logger.info(f"Acc === {acc.item()}")

            rewards_1, rewards_2 = episode.get_rewards(logits)
            logger.info(f"MEDIAN REWARD A1 === {rewards_1.mean().item()}")
            logger.info(f"MEDIAN REWARD A2 === {rewards_2.mean().item()}")

            self.train_op_1
            self.train_op_2

            if self.batch_counter == self.params['rounds_sub_training']:
                torch.save(self.agent.state_dict(), os.path.join(
                    self.params['model_dir'], 'unbiased_model.pt'))
                torch.save(self.judge.state_dict(), os.path.join(
                    self.params['model_dir'], 'unbiased_model.pt'))

            if self.batch_counter % self.params['eval_every'] == 0:
                self.test(True)

            if self.batch_counter >= self.params['total_iterations']:
                break


def test(self, is_dev_environment, save_model=False, best_threshold=None):
    batch_counter = 0
    total_examples = 0
    mean_probs_list = []
    correct_answer_list = []
    sk_mean_logit_list = []
    sk_correct_answer_list = []
    hitsAt20 = 0
    hitsAt10 = 0
    hitsAt3 = 0
    hitsAt1 = 0
    mean_reciprocal_rank = 0
    mean_rank = 0
    debate_printer = Debate_Printer(
        self.params['output_dir'], self.train_environment.grapher, self.params['test_rollouts'], is_append=True)

    environment = self.dev_test_environment if is_dev_environment else self.test_test_environment
    for episode in tqdm(environment.get_episodes()):
        batch_counter += 1

        episode_answers = episode.get_labels()

        self.query_subject.copy_(episode.get_query_subjects())
        self.query_relation.copy_(episode.get_query_relation())
        self.query_object.copy_(episode.get_query_objects())
        self.labels.copy_(episode_answers)

        agent_mem_1, agent_mem_2 = self.agent.get_init_state_array(
            episode.no_examples)
        previous_relation = torch.ones(
            episode.no_examples * self.params['test_rollouts'], dtype=torch.long) * self.params['relation_vocab']['DUMMY_START_RELATION']

        debate_printer.create_debates(episode.get_query_subjects(
        ), episode.get_query_relation(), episode.get_query_objects(), episode_answers)

        input_argument = 0
        debate_printer_rel_list = []
        debate_printer_ent_list = []
        rep_argu_list = []  # Defined the list to store the hidden representations
        for argument_num in range(self.params['number_arguments']):
            state = episode.reset_initial_state()
            for path_num in range(self.params['path_length']):
                loss, agent_mem_1, agent_mem_2, test_scores, test_action_idx, chosen_relation, input_argument = self.agent(
                    state['next_relations'], state['next_entities'], agent_mem_1, agent_mem_2, previous_relation, state['current_entities'], self.range_arr, 0.0, self.random_flag)
                previous_relation = chosen_relation
                rel_string, ent_string = debate_printer.get_action_rel_ent(
                    test_action_idx, state)
                debate_printer_rel_list.append(rel_string)
                debate_printer_ent_list.append(ent_string)
                state = episode(test_action_idx)

            logits_last_argument, hidden_rep_argu = self.judge(input_argument)
            rewards = logits_last_argument
            debate_printer.create_arguments(
                debate_printer_rel_list, debate_printer_ent_list, rewards, True)
            debate_printer_rel_list.clear()
            debate_printer_ent_list.clear()
            rep_argu_list.append(hidden_rep_argu)

            mean_argu_rep = torch.mean(torch.stack(
                [torch.unsqueeze(rep_argu, -1) for rep_argu in rep_argu_list], -1), -1)
            logits_judge = self.judge(mean_argu_rep)

            reshaped_logits_judge = logits_judge.reshape(
                episode.no_examples, self.params['test_rollouts'])
            reshaped_answer = episode_answers.reshape(
                episode.no_examples, self.params['test_rollouts'])
            correct_answer_list.append(reshaped_answer[:, 0])
            probs_judge = torch.sigmoid(reshaped_logits_judge)
            mean_probs = torch.mean(probs_judge, dim=1, keepdim=True)
            mean_probs_list.append(mean_probs)
            reshaped_mean_probs = mean_probs.reshape(1, -1)
            idx_final_logits = torch.argsort(
                reshaped_mean_probs, dim=1, descending=True)

            mean_logits = torch.mean(reshaped_logits_judge, dim=1)
            sk_mean_logit_list.append(mean_logits)
            sk_correct_answer_list.append(reshaped_answer[:, 0])

            debate_printer.create_best_debates()
            debate_printer.set_debates_final_logit(mean_logits)
            debate_printer.write('argument_test_new.txt')

            for fact in idx_final_logits[0]:
                ans_rank = None
                rank = 0
                for ix in reversed(idx_final_logits[0]):
                    if ix == 0:
                        ans_rank = rank
                        break
                    rank += 1
                mean_reciprocal_rank += 1 / \
                    (ans_rank+1) if ans_rank is not None else 0
                mean_rank += ans_rank + \
                    1 if ans_rank is not None else reshaped_mean_probs.size(
                        1) + 1
                if rank < 20:
                    hitsAt20 += 1
                    if rank < 10:
                        hitsAt10 += 1
                        if rank < 3:
                            hitsAt3 += 1
                            if rank == 0:
                                hitsAt1 += 1

            total_examples += reshaped_mean_probs.size(1)

        sk_correct_answer_list = torch.cat(
            sk_correct_answer_list).to(torch.long)
        sk_mean_logit_list = torch.cat(sk_mean_logit_list)
        precision, recall, thresholds = self.get_precision_recall_curve(
            sk_correct_answer_list, sk_mean_logit_list)
        auc_pr = self.get_auc(recall, precision)
        fpr, tpr, thresholds = self.get_roc_curve(
            sk_correct_answer_list, sk_mean_logit_list)
        auc_roc = self.get_auc(fpr, tpr)

        best_acc = -1
        if best_threshold is None:
            for threshold in thresholds:
                binary_preds = (sk_mean_logit_list > threshold).to(torch.long)
                acc = self.get_accuracy(binary_preds, sk_correct_answer_list)
                if best_acc < acc:
                    best_acc = acc
                    best_threshold = threshold
        else:
            for threshold in thresholds:
                binary_preds = (sk_mean_logit_list > threshold).to(torch.long)
                acc = self.get_accuracy(binary_preds, sk_correct_answer_list)
                if best_acc < acc:
                    best_acc = acc
                    wrong_best_threshold = threshold

            logger.info(f"NOT BEST ACC === {best_acc}")
            logger.info(f"NOT BEST THRESHOLD === {wrong_best_threshold}")
            binary_preds = (sk_mean_logit_list > best_threshold).to(torch.long)
            best_acc = self.get_accuracy(binary_preds, sk_correct_answer_list)

        logger.info("========== SKLEARN METRICS =============")
        logger.info(f"Best Threshold === {best_threshold}")
        logger.info(f"Acc === {best_acc}")
        logger.info(f"AUC_PR === {auc_pr}")
        logger.info(f"AUC_ROC === {auc_roc}")
        logger.info("========================================")

        if self.params['is_use_fixed_false_facts']:
            if save_model or best_acc > self.best_metric:
                torch.save(self.agent.state_dict(), os.path.join(
                    self.params['model_dir'], "model.pt"))
                torch.save(self.judge.state_dict(), os.path.join(
                    self.params['model_dir'], "model.pt"))
            self.best_metric = best_acc if best_acc > self.best_metric else self.best_metric
            self.best_threshold = best_threshold
        else:
            if save_model or mean_reciprocal_rank > self.best_metric:
                torch.save(self.agent.state_dict(), os.path.join(
                    self.params['model_dir'], "model.pt"))
                torch.save(self.judge.state_dict(), os.path.join(
                    self.params['model_dir'], "model.pt"))
            self.best_metric = mean_reciprocal_rank if mean_reciprocal_rank > self.best_metric else self.best_metric

        logger.info(f"Hits@20 === {hitsAt20 / total_examples}")
        logger.info(f"Hits@10 === {hitsAt10 / total_examples}")
        logger.info(f"Hits@3 === {hitsAt3 / total_examples}")
        logger.info(f"Hits@1 === {hitsAt1 / total_examples}")
        logger.info(f"MRR === {mean_reciprocal_rank / total_examples}")
        logger.info(f"MR === {mean_rank / total_examples}")

        return best_threshold

    def get_precision_recall_curve(self, y_true, y_score):
        return sklearn.metrics.precision_recall_curve(y_true, y_score)

    def get_roc_curve(self, y_true, y_score):
        return sklearn.metrics.roc_curve(y_true, y_score)

    def get_auc(self, x, y):
        return sklearn.metrics.auc(x, y)

    def get_accuracy(self, y_pred, y_true):
        return sklearn.metrics.accuracy_score(y_true, y_pred)


def main():
    option = read_options()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = None
    logger.addHandler(console)

    relation_vocab = json.load(
        open(option['vocab_dir'] + '/relation_vocab.json'))
    entity_vocab = json.load(open(option['vocab_dir'] + '/entity_vocab.json'))
    mid_to_name = json.load(open(option['vocab_dir'] + '/fb15k_names.json')
                            ) if os.path.isfile(option['vocab_dir'] + '/fb15k_names.json') else None

    logger.info(f'Total number of entities {len(entity_vocab)}')
    logger.info(f'Total number of relations {len(relation_vocab)}')

    if not option['load_model']:
        for key, val in option.items():
            if not isinstance(val, list):
                option[key] = [val]

        for permutation in ParameterGrid(option):
            best_permutation = None
            best_metric = 0

            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
            permutation['output_dir'] = permutation['base_output_dir'] + '/' + str(current_time) + '__' + str(uuid.uuid4())[:4] + '_' + str(
                permutation['path_length']) + '_' + str(permutation['beta']) + '_' + str(permutation['test_rollouts']) + '_' + str(permutation['Lambda'])
            permutation['model_dir'] = os.path.join(
                permutation['output_dir'], 'model')
            permutation['load_model'] = (permutation['load_model'] == 1)

            os.makedirs(permutation['output_dir'])
            os.mkdir(permutation['model_dir'])
            with open(os.path.join(permutation['output_dir'], 'config.txt'), 'w') as out:
                pprint(permutation, stream=out)

            print('Arguments:')
            maxLen = max([len(ii) for ii in permutation.keys()])
            fmtString = '\t%' + str(maxLen) + 's : %s'
            for keyPair in sorted(permutation.items()):
                print(fmtString % keyPair)

            logger.removeHandler(logfile)
            logfile = logging.FileHandler(os.path.join(
                permutation['output_dir'], 'log.txt'), 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)

            permutation['relation_vocab'] = relation_vocab
            permutation['entity_vocab'] = entity_vocab
            permutation['mid_to_name'] = mid_to_name

            trainer = Trainer(permutation, best_metric)
            trainer.set_random_seeds(permutation.get('seed', None))
            trainer.initialize()
            trainer.train()

            if trainer.best_metric > best_metric or best_permutation is None:
                best_acc = trainer.best_metric
                best_threshold = trainer.best_threshold
                best_permutation = permutation
            torch.cuda.empty_cache()

        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        best_permutation['output_dir'] = best_permutation['base_output_dir'] + '/' + str(current_time) + '__Test__' + str(uuid.uuid4())[:4] + '_' + str(
            best_permutation['path_length']) + '_' + str(best_permutation['beta']) + '_' + str(best_permutation['test_rollouts']) + '_' + str(best_permutation['Lambda'])
        best_permutation['old_model_dir'] = best_permutation['model_dir']
        best_permutation['model_dir'] = os.path.join(
            best_permutation['output_dir'], 'model')
        best_permutation['load_model'] = (best_permutation['load_model'] == 1)

        os.makedirs(best_permutation['output_dir'])
        os.mkdir(best_permutation['model_dir'])
        with open(os.path.join(best_permutation['output_dir'], 'config.txt'), 'w') as out:
            pprint(best_permutation, stream=out)

        print('Arguments:')
        maxLen = max([len(ii) for ii in best_permutation.keys()])
        fmtString = '\t%' + str(maxLen) + 's : %s'
        for keyPair in sorted(best_permutation.items()):
            if not keyPair[0].endswith('_vocab') and keyPair[0] != 'mid_to_name':
                print(fmtString % keyPair)

        logger.removeHandler(logfile)
        logfile = logging.FileHandler(os.path.join(
            best_permutation['output_dir'], 'log.txt'), 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

        trainer = Trainer(best_permutation, best_acc)
        trainer.set_random_seeds(best_permutation.get('seed', None))
        trainer.initialize(os.path.join(
            best_permutation['old_model_dir'], "model.pt"))
        trainer.test(False, True, best_threshold)
    else:
        logger.info("Skipping training")
        logger.info(f"Loading model from {option['model_load_dir']}")

        for key, value in option.items():
            if isinstance(value, list):
                if len(value) == 1:
                    option[key] = value[0]
                else:
                    raise ValueError(
                        f"Parameter {key} has more than one value in the config file.")

        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        option['output_dir'] = option['base_output_dir'] + '/' + str(current_time) + '__Test__' + str(uuid.uuid4())[:4] + '_' + str(
            option['path_length']) + '_' + str(option['beta']) + '_' + str(option['test_rollouts']) + '_' + str(option['Lambda'])
        option['model_dir'] = os.path.join(option['output_dir'], 'model')
        os.makedirs(option['output_dir'])
        os.mkdir(option['model_dir'])
        with open(os.path.join(option['output_dir'], 'config.txt'), 'w') as out:
            pprint(option, stream=out)

        logger.removeHandler(logfile)
        logfile = logging.FileHandler(os.path.join(
            option['output_dir'], 'log.txt'), 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

        option['relation_vocab'] = relation_vocab
        option['entity_vocab'] = entity_vocab
        option['mid_to_name'] = mid_to_name

        trainer = Trainer(option, 0)
        trainer.set_random_seeds(option.get('seed', None))
        trainer.initialize(option['model_load_dir'] + "model.pt")
        if option['is_use_fixed_false_facts']:
            best_threshold = trainer.test(True, False)
            trainer.test(False, True, best_threshold=best_threshold)
        else:
            trainer.test(False, True)


if __name__ == '__main__':
    main()

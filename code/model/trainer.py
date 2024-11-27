from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import pickle
import os
import uuid
import datetime
from pprint import pprint
import logging
import numpy as np
from scipy.special import expit as sig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from code.model.judge import Judge
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
from sklearn.model_selection import ParameterGrid
import itertools
import random
import sklearn
import gc

if os.name == 'posix':
    import resource
import sys
from code.model.baseline import ReactiveBaseline
from code.model.debate_printer import Debate_Printer

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Trainer:
    '''
    Central class for R2D2 implemented in PyTorch. The trainer handles all components of the model and manages training/testing.
    '''
    
    def __init__(self, params, best_metric):
        '''
        Initialize the trainer with given parameters and best metric.
        '''
        # Transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)
        
        self.set_random_seeds(self.seed)
        self.batch_size = self.batch_size * (1 + self.false_facts_train) * self.num_rollouts
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.judge = Judge(params).to(self.device)
        self.agent = Agent(params, self.judge).to(self.device)
        
        # Initialize environments
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        
        self.number_steps = self.path_length * self.number_arguments * 2
        self.best_metric = best_metric

        # Optimization setup
        self.learning_rate_judge_init = params['learning_rate_judge']
        self.baseline_1 = ReactiveBaseline(l=self.Lambda)
        self.baseline_2 = ReactiveBaseline(l=self.Lambda)
        
        # Initialize optimizers
        self.optimizer_judge = Adam(self.judge.parameters(), lr=self.learning_rate_judge_init)
        self.optimizer_agents = Adam(
            list(self.agent.policy_agent_1_cells.parameters()) + 
            list(self.agent.policy_agent_2_cells.parameters()), 
            lr=self.learning_rate_agents
        )

    def set_random_seeds(self, seed):
        '''Set random seeds for reproducibility'''
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

    def calc_reinforce_loss(self, per_example_loss, which_agent_sequence, 
                          cum_discounted_reward_1, cum_discounted_reward_2, per_example_logits):
        '''Calculate REINFORCE loss for both agents'''
        
        # Stack losses and create masks
        loss = torch.stack(per_example_loss, dim=1)  # [B, T]
        mask = torch.tensor(which_agent_sequence).bool().to(self.device)
        mask = mask.expand(loss.size(0), -1)
        
        # Split losses by agent
        loss_1 = loss.masked_select(~mask).view(loss.size(0), -1)
        loss_2 = loss.masked_select(mask).view(loss.size(0), -1)

        # Calculate rewards
        if self.custom_baseline:
            final_reward_1 = cum_discounted_reward_1
            final_reward_2 = cum_discounted_reward_2
        else:
            baseline_1 = self.baseline_1.get_baseline_value()
            baseline_2 = self.baseline_2.get_baseline_value()
            
            final_reward_1 = cum_discounted_reward_1 - baseline_1
            final_reward_2 = cum_discounted_reward_2 - baseline_2

        # Normalize rewards
        final_reward_1 = (final_reward_1 - final_reward_1.mean()) / (final_reward_1.std() + 1e-6)
        final_reward_2 = (final_reward_2 - final_reward_2.mean()) / (final_reward_2.std() + 1e-6)

        # Calculate losses with rewards
        loss_1 = torch.mul(loss_1, final_reward_1)
        loss_2 = torch.mul(loss_2, final_reward_2)

        # Add entropy regularization
        entropy_1, entropy_2 = self.entropy_reg_loss(per_example_logits, which_agent_sequence)
        total_loss_1 = loss_1.mean() - self.decaying_beta * entropy_1
        total_loss_2 = loss_2.mean() - self.decaying_beta * entropy_2

        return total_loss_1, total_loss_2

    def entropy_reg_loss(self, all_logits, which_agent_sequence):
        '''Calculate entropy regularization loss'''
        logits = torch.stack(all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        mask = torch.tensor(which_agent_sequence).bool().to(self.device)
        mask = mask.expand(logits.size(0), logits.size(1), -1)
        
        # Split logits by agent
        logits_1 = logits.masked_select(~mask).view(logits.size(0), logits.size(1), -1)
        logits_2 = logits.masked_select(mask).view(logits.size(0), logits.size(1), -1)
        
        # Calculate entropy
        entropy_1 = -(F.softmax(logits_1, dim=1) * F.log_softmax(logits_1, dim=1)).sum(dim=1).mean()
        entropy_2 = -(F.softmax(logits_2, dim=1) * F.log_softmax(logits_2, dim=1)).sum(dim=1).mean()
        
        return entropy_1, entropy_2

    def train(self, epochs):
        '''Main training loop'''
        self.batch_counter = 0
        
        debate_printer = Debate_Printer(self.output_dir, self.train_environment.grapher, self.num_rollouts)
        
        for epoch in range(epochs):
            for episode in self.train_environment.get_episodes():
                # Determine if we're training judge or agents
                is_train_judge = (self.batch_counter // self.train_judge_every) % 2 == 0 \
                                or self.batch_counter >= self.rounds_sub_training
                
                logger.info(f"BATCH COUNTER: {self.batch_counter}")
                self.batch_counter += 1

                # Get episode data
                query_subjects = torch.tensor(episode.get_query_subjects()).to(self.device)
                query_relation = torch.tensor(episode.get_query_relation()).to(self.device)
                query_objects = torch.tensor(episode.get_query_objects()).to(self.device)
                episode_answers = torch.tensor(episode.get_labels()).to(self.device)

                # Set embeddings
                self.judge.set_query_embeddings(query_subjects, query_relation, query_objects)
                self.agent.set_query_embeddings(query_subjects, query_relation, query_objects)

                debate_printer.create_debates(
                    episode.get_query_subjects(),
                    episode.get_query_relation(),
                    episode.get_query_objects(),
                    episode.get_labels()
                )

                # Training loop for each argument
                loss_before_regularization = []
                logits = []
                which_agent_list = []
                per_example_loss = []
                per_example_logits = []
                action_idx = []
                rewards_agents = []
                rewards_before_baseline = []
                
                for arguments in range(self.number_arguments):
                    # Pro agent's turn
                    state = episode.reset_initial_state()
                    for path_num in range(self.path_length):
                        which_agent_list.append(0.0)
                        loss, logits_out, idx, rewards, before_baseline = self.agent_step(
                            state, 0, query_subjects, query_relation, query_objects
                        )
                        loss_before_regularization.append(loss)
                        per_example_loss.append(loss)
                        per_example_logits.append(logits_out)
                        action_idx.append(idx)
                        rewards_agents.append(rewards)
                        rewards_before_baseline.append(before_baseline)
                        state = episode(idx)
                        logits.append((0, rewards))

                    # Con agent's turn
                    state = episode.reset_initial_state()
                    for path_num in range(self.path_length):
                        which_agent_list.append(1.0)
                        loss, logits_out, idx, rewards, before_baseline = self.agent_step(
                            state, 1, query_subjects, query_relation, query_objects
                        )
                        loss_before_regularization.append(loss)
                        per_example_loss.append(loss)
                        per_example_logits.append(logits_out)
                        action_idx.append(idx)
                        rewards_agents.append(rewards)
                        rewards_before_baseline.append(before_baseline)
                        state = episode(idx)
                        logits.append((1, rewards))

                # Calculate final judge outputs
                judge_outputs = self.judge(logits)
                logits_judge = judge_outputs['final_logits']
                debate_printer.set_debates_final_logit(logits_judge.detach().cpu().numpy())

                # Training step
                if is_train_judge:
                    loss_judge = judge_outputs['loss']
                    self.optimizer_judge.zero_grad()
                    loss_judge.backward()
                    torch.nn.utils.clip_grad_norm_(self.judge.parameters(), self.grad_clip_norm)
                    self.optimizer_judge.step()
                else:
                    # Train agents
                    rewards_1, rewards_2 = episode.get_rewards(logits)
                    rewards_1 = torch.tensor(rewards_1).to(self.device)
                    rewards_2 = torch.tensor(rewards_2).to(self.device)
                    
                    cum_discounted_reward_1, cum_discounted_reward_2 = self.calc_cum_discounted_reward(
                        rewards_1, rewards_2, np.array(which_agent_list)
                    )
                    
                    loss_1, loss_2 = self.calc_reinforce_loss(
                        per_example_loss,
                        which_agent_list,
                        cum_discounted_reward_1,
                        cum_discounted_reward_2,
                        per_example_logits
                    )
                    
                    self.optimizer_agents.zero_grad()
                    (loss_1 + loss_2).backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.agent.policy_agent_1_cells.parameters()) + 
                        list(self.agent.policy_agent_2_cells.parameters()),
                        self.grad_clip_norm
                    )
                    self.optimizer_agents.step()

                # Evaluation and checkpointing
                if self.batch_counter % self.eval_every == 0:
                    self.test(True)

                if self.batch_counter == self.rounds_sub_training:
                    torch.save({
                        'model_state_dict': self.agent.state_dict(),
                        'judge_state_dict': self.judge.state_dict(),
                        'optimizer_agents': self.optimizer_agents.state_dict(),
                        'optimizer_judge': self.optimizer_judge.state_dict(),
                    }, f"{self.model_dir}/unbiased_model/unbiased_model.pt")

                if os.name == 'posix':
                    logger.info('Memory usage : %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                gc.collect()
                if self.batch_counter >= self.total_iterations:
                    return

    def agent_step(self, state, which_agent, query_subjects, query_relation, query_objects):
        '''Execute a single step for an agent'''
        next_relations = torch.tensor(state['next_relations']).to(self.device)
        next_entities = torch.tensor(state['next_entities']).to(self.device)
        current_entities = torch.tensor(state['current_entities']).to(self.device)
        
        outputs = self.agent(
            which_agent=which_agent,
            next_relations=next_relations,
            next_entities=next_entities,
            current_entities=current_entities,
            query_subject=query_subjects,
            query_relation=query_relation,
            query_object=query_objects
        )
        
        return outputs['loss'], outputs['logits'], outputs['action_idx'], \
               outputs['rewards'], outputs['rewards_before_baseline']

    def test(self, is_dev_environment, save_model=False, best_threshold=None):
        '''Test the model'''
        self.agent.eval()
        self.judge.eval()
        
        total_examples = 0
        mean_probs_list = []
        correct_answer_list = []
        environment = self.dev_test_environment if is_dev_environment else self.test_test_environment
        
        metrics = {
            'hits@20': 0,
            'hits@10': 0,
            'hits@3': 0,
            'hits@1': 0,
            'mrr': 0,
            'mr': 0
        }
        
        with torch.no_grad():
            for episode in tqdm(environment.get_episodes()):
                # Process episode
                query_subjects = torch.tensor(episode.get_query_subjects()).to(self.device)
                query_relation = torch.tensor(episode.get_query_relation()).to(self.device)
                query_objects = torch.tensor(episode.get_query_objects()).to(self.device)
                episode_answers = torch.tensor(episode.get_labels()).to(self.device)
                
                # Test episode
                logits_judge = self.test_episode(
                    episode, query_subjects, query_relation, query_objects, episode_answers
                )
                
                # Update metrics
                probs_judge = torch.sigmoid(logits_judge)
                mean_probs = probs_judge.mean(dim=1, keepdim=True)
                mean_probs_list.append(mean_probs)
                correct_answer_list.append(episode_answers[:, [0]])

                # Calculate rankings and metrics
                reshaped_mean_probs = mean_probs.reshape(
                    -1, episode.no_examples)
                idx_final_logits = torch.argsort(reshaped_mean_probs, dim=1)

                for fact in idx_final_logits:
                    ans_rank = None
                    rank = 0
                    for ix in reversed(fact):
                        if ix == 0:
                            ans_rank = rank
                            break
                        rank += 1

                    if ans_rank is not None:
                        metrics['mrr'] += 1 / (ans_rank + 1)
                        metrics['mr'] += ans_rank + 1
                        if rank < 20:
                            metrics['hits@20'] += 1
                            if rank < 10:
                                metrics['hits@10'] += 1
                                if rank < 3:
                                    metrics['hits@3'] += 1
                                    if rank == 0:
                                        metrics['hits@1'] += 1
                    else:
                        metrics['mr'] += reshaped_mean_probs.shape[1] + 1

                total_examples += reshaped_mean_probs.shape[0]

        # Calculate final metrics
        mean_probs_tensor = torch.cat(mean_probs_list).cpu().numpy()
        correct_answers_tensor = torch.cat(
            correct_answer_list).cpu().numpy().astype(int)

        # Calculate AUC-PR and AUC-ROC
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            correct_answers_tensor, mean_probs_tensor)
        auc_pr = sklearn.metrics.auc(recall, precision)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            correct_answers_tensor, mean_probs_tensor)
        auc_roc = sklearn.metrics.auc(fpr, tpr)

        # Find best threshold and accuracy
        if best_threshold is None:
            best_acc = -1
            best_threshold = 0
            for threshold in thresholds:
                binary_preds = (mean_probs_tensor > threshold).astype(int)
                acc = sklearn.metrics.accuracy_score(
                    binary_preds, correct_answers_tensor)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
        else:
            best_acc = -1
            wrong_best_threshold = 0
            for threshold in thresholds:
                binary_preds = (mean_probs_tensor > threshold).astype(int)
                acc = sklearn.metrics.accuracy_score(
                    binary_preds, correct_answers_tensor)
                if acc > best_acc:
                    best_acc = acc
                    wrong_best_threshold = threshold

            logger.info(f"NOT BEST ACC === {best_acc}")
            logger.info(f"NOT BEST THRESHOLD === {wrong_best_threshold}")
            binary_preds = (mean_probs_tensor > best_threshold).astype(int)
            best_acc = sklearn.metrics.accuracy_score(
                binary_preds, correct_answers_tensor)

        # Log metrics
        logger.info("========== SKLEARN METRICS =============")
        logger.info(f"Best Threshold === {best_threshold}")
        logger.info(f"Acc === {best_acc}")
        logger.info(f"AUC_PR === {auc_pr}")
        logger.info(f"AUC_ROC === {auc_roc}")
        logger.info("========================================")

        # Save best model
        if self.is_use_fixed_false_facts:
            if save_model or best_acc > self.best_metric:
                torch.save({
                    'model_state_dict': self.agent.state_dict(),
                    'judge_state_dict': self.judge.state_dict(),
                    'optimizer_agents': self.optimizer_agents.state_dict(),
                    'optimizer_judge': self.optimizer_judge.state_dict(),
                }, f"{self.model_dir}/model.pt")
            self.best_metric = max(best_acc, self.best_metric)
            self.best_threshold = best_threshold
        else:
            if save_model or metrics['mrr']/total_examples > self.best_metric:
                torch.save({
                    'model_state_dict': self.agent.state_dict(),
                    'judge_state_dict': self.judge.state_dict(),
                    'optimizer_agents': self.optimizer_agents.state_dict(),
                    'optimizer_judge': self.optimizer_judge.state_dict(),
                }, f"{self.model_dir}/model.pt")
            self.best_metric = max(
                metrics['mrr']/total_examples, self.best_metric)

        # Log ranking metrics
        logger.info(f"Hits@20 === {metrics['hits@20'] / total_examples}")
        logger.info(f"Hits@10 === {metrics['hits@10'] / total_examples}")
        logger.info(f"Hits@3 === {metrics['hits@3'] / total_examples}")
        logger.info(f"Hits@1 === {metrics['hits@1'] / total_examples}")
        logger.info(f"MRR === {metrics['mrr'] / total_examples}")
        logger.info(f"MR === {metrics['mr'] / total_examples}")

        # Restore training mode
        self.agent.train()
        self.judge.train()

        return best_threshold

    def test_episode(self, episode, query_subjects, query_relation, query_objects, episode_answers):
        '''Test a single episode'''
        temp_batch_size = episode.no_examples
        debate_printer = Debate_Printer(
            self.output_dir,
            self.train_environment.grapher,
            self.test_rollouts,
            is_append=True
        )

        # Initialize agent states
        agent_mem_1, agent_mem_2 = self.agent.get_init_state_array(
            temp_batch_size)

        # Process arguments
        logits = []
        rep_argu_list = []

        debate_printer.create_debates(
            episode.get_query_subjects(),
            episode.get_query_relation(),
            episode.get_query_objects(),
            episode.get_labels()
        )

        input_argument = torch.zeros(1)  # Dummy initial value
        debate_printer_rel_list = []
        debate_printer_ent_list = []

        for argument_num in range(self.number_arguments):
            # Pro agent's turn
            state = episode.reset_initial_state()
            for path_num in range(self.path_length):
                state_tensors = {
                    'next_relations': torch.tensor(state['next_relations']).to(self.device),
                    'next_entities': torch.tensor(state['next_entities']).to(self.device),
                    'current_entities': torch.tensor(state['current_entities']).to(self.device)
                }

                outputs = self.agent.step_test(
                    state_tensors,
                    which_agent=0,
                    agent_mem_1=agent_mem_1,
                    agent_mem_2=agent_mem_2,
                    input_argument=input_argument
                )

                agent_mem_1 = outputs['agent_mem_1']
                agent_mem_2 = outputs['agent_mem_2']
                action_idx = outputs['action_idx']
                input_argument = outputs['new_argument']

                state = episode(action_idx.cpu().numpy())

                rel_string, ent_string = debate_printer.get_action_rel_ent(
                    action_idx.cpu().numpy(), state)
                debate_printer_rel_list.append(rel_string)
                debate_printer_ent_list.append(ent_string)

            # Get argument logits and representation
            arg_outputs = self.judge.classify_argument(input_argument)
            logits_last_argument = arg_outputs['logits']
            hidden_rep_argu = arg_outputs['hidden']

            rewards = logits_last_argument
            rep_argu_list.append(hidden_rep_argu)
            debate_printer.create_arguments(
                debate_printer_rel_list, debate_printer_ent_list, rewards.cpu().numpy(), True)

            debate_printer_rel_list.clear()
            debate_printer_ent_list.clear()

            # Con agent's turn
            state = episode.reset_initial_state()
            for path_num in range(self.path_length):
                state_tensors = {
                    'next_relations': torch.tensor(state['next_relations']).to(self.device),
                    'next_entities': torch.tensor(state['next_entities']).to(self.device),
                    'current_entities': torch.tensor(state['current_entities']).to(self.device)
                }

                outputs = self.agent.step_test(
                    state_tensors,
                    which_agent=1,
                    agent_mem_1=agent_mem_1,
                    agent_mem_2=agent_mem_2,
                    input_argument=input_argument
                )

                agent_mem_1 = outputs['agent_mem_1']
                agent_mem_2 = outputs['agent_mem_2']
                action_idx = outputs['action_idx']
                input_argument = outputs['new_argument']

                state = episode(action_idx.cpu().numpy())

                rel_string, ent_string = debate_printer.get_action_rel_ent(
                    action_idx.cpu().numpy(), state)
                debate_printer_rel_list.append(rel_string)
                debate_printer_ent_list.append(ent_string)

            # Get argument logits and representation
            arg_outputs = self.judge.classify_argument(input_argument)
            logits_last_argument = arg_outputs['logits']
            hidden_rep_argu = arg_outputs['hidden']

            rewards = logits_last_argument
            rep_argu_list.append(hidden_rep_argu)
            debate_printer.create_arguments(
                debate_printer_rel_list, debate_printer_ent_list, rewards.cpu().numpy(), False)

            debate_printer_rel_list.clear()
            debate_printer_ent_list.clear()

        # Calculate final judge logits
        mean_argu_rep = torch.stack(rep_argu_list, dim=-1).mean(dim=-1)
        logits_judge = self.judge.get_logits_argument(mean_argu_rep)

        debate_printer.create_best_debates()
        debate_printer.set_debates_final_logit(logits_judge.cpu().numpy())
        debate_printer.write('argument_test_new.txt')

        return logits_judge

    def calc_cum_discounted_reward(self, rewards_1, rewards_2, which_agent_sequence):
        '''Calculate cumulative discounted rewards for both agents'''
        r_1_index = -1
        r_2_index = -1
        running_add_1 = torch.zeros(rewards_1.shape[0], device=self.device)
        no_paths_1 = (which_agent_sequence == 0.0).sum()
        cum_disc_reward_1 = torch.zeros(
            rewards_1.shape[0],
            no_paths_1,
            device=self.device
        )

        running_add_2 = torch.zeros(rewards_2.shape[0], device=self.device)
        no_paths_2 = (which_agent_sequence == 1.0).sum()
        cum_disc_reward_2 = torch.zeros(
            rewards_2.shape[0],
            no_paths_2,
            device=self.device
        )

        prev_t = None
        for t in reversed(which_agent_sequence):
            if t == 0.0:
                if prev_t == 0:
                    running_add_1 = self.gamma * \
                        running_add_1 + rewards_1[:, r_1_index]
                else:
                    running_add_1 = rewards_1[:, r_1_index]
                cum_disc_reward_1[:, r_1_index] = running_add_1
                r_1_index -= 1
            if t == 1.0:
                if prev_t == 1:
                    running_add_2 = self.gamma * \
                        running_add_2 + rewards_2[:, r_2_index]
                else:
                    running_add_2 = rewards_2[:, r_2_index]
                cum_disc_reward_2[:, r_2_index] = running_add_2
                r_2_index -= 1
            prev_t = t

        return cum_disc_reward_1, cum_disc_reward_2


def main():
    '''Main function to run training or evaluation'''
    option = read_options()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = None
    logger.addHandler(console)

    # Read vocabulary files
    logger.info('Reading vocab files...')
    relation_vocab = json.load(
        open(option['vocab_dir'] + '/relation_vocab.json'))
    entity_vocab = json.load(open(option['vocab_dir'] + '/entity_vocab.json'))
    mid_to_name = json.load(open(option['vocab_dir'] + '/fb15k_names.json')) \
        if os.path.isfile(option['vocab_dir'] + '/fb15k_names.json') else None

    logger.info('Reading mid to name map')
    logger.info('Done..')
    logger.info(f'Total number of entities {len(entity_vocab)}')
    logger.info(f'Total number of relations {len(relation_vocab)}')

    if not option['load_model']:
        # Training mode
        for key, val in option.items():
            if not isinstance(val, list):
                option[key] = [val]

        for permutation in ParameterGrid(option):
            best_permutation = None
            best_metric = 0

            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
            # Setup output directories and logging
            permutation['output_dir'] = setup_directories(
                permutation, current_time)
            setup_logging(permutation, fmt)

            # Initialize and train model
            trainer = Trainer(permutation, best_metric)
            trainer.train(permutation['epochs'])

            if trainer.best_metric > best_metric or best_permutation is None:
                best_acc = trainer.best_metric
                best_threshold = trainer.best_threshold
                best_permutation = permutation

    else:
        # Evaluation mode
        logger.info("Skipping training")
        logger.info(f"Loading model from {option['model_load_dir']}")

        # Setup directories and logging for evaluation
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        option['output_dir'] = setup_directories(
            option, current_time, is_eval=True)
        setup_logging(option, fmt)

        trainer = Trainer(option, 0)

        # Load saved model
        checkpoint = torch.load(f"{option['model_load_dir']}/model.pt")
        trainer.agent.load_state_dict(checkpoint['model_state_dict'])
        trainer.judge.load_state_dict(checkpoint['judge_state_dict'])
        trainer.optimizer_agents.load_state_dict(
            checkpoint['optimizer_agents'])
        trainer.optimizer_judge.load_state_dict(checkpoint['optimizer_judge'])

        # Run evaluation
        if option['is_use_fixed_false_facts']:
            best_threshold = trainer.test(True, False)
            trainer.test(False, True, best_threshold=best_threshold)
        else:
            trainer.test(False, True)


def setup_directories(params, current_time, is_eval=False):
    '''Setup output directories for training or evaluation'''
    if is_eval:
        output_dir = params['base_output_dir'] + '/' + current_time + '__Test__' + \
            str(uuid.uuid4())[:4] + '_' + str(params['path_length']) + '_' + \
            str(params['beta']) + '_' + str(params['test_rollouts']) + '_' + \
            str(params['Lambda'])
    else:
        output_dir = params['base_output_dir'] + '/' + current_time + '__' + \
            str(uuid.uuid4())[:4] + '_' + str(params['path_length']) + '_' + \
            str(params['beta']) + '_' + str(params['test_rollouts']) + '_' + \
            str(params['Lambda'])

    # Create output and model directories
    model_dir = output_dir + '/model/'
    os.makedirs(output_dir)
    os.makedirs(model_dir)

    # Save configuration
    with open(output_dir + '/config.txt', 'w') as out:
        pprint(params, stream=out)

    params['output_dir'] = output_dir
    params['model_dir'] = model_dir

    return output_dir


def setup_logging(params, fmt):
    '''Setup logging configuration'''
    # Remove existing handlers
    logger.handlers.clear()

    # Add file handler
    logfile = logging.FileHandler(params['output_dir'] + '/log.txt', 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # Add console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Print parameters
    maxLen = max([len(str(key)) for key in params.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(params.items()):
        if not str(keyPair[0]).endswith('_vocab') and keyPair[0] != 'mid_to_name':
            print(fmtString % keyPair)


if __name__ == '__main__':
    main()

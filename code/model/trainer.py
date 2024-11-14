import json
from sklearn.model_selection import ParameterGrid
from pprint import pprint
import datetime
import uuid
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import random

from code.model.judge import Judge
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
from code.model.baseline import ReactiveBaseline
from code.model.debate_printer import Debate_Printer

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


class Trainer:
    def __init__(self, params, best_metric):
        # Transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        self.set_random_seeds(self.seed)

        self.batch_size *= (1 + self.false_facts_train) * self.num_rollouts
        self.judge = Judge(params)
        self.agent = Agent(params, self.judge)
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.number_steps = self.path_length * self.number_arguments * 2
        self.best_metric = best_metric

        self.learning_rate_judge = params['learning_rate_judge']
        self.optimizer_judge = optim.Adam(
            self.judge.parameters(), lr=self.learning_rate_judge)
        self.optimizer_agents = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate_agents)

        self.baseline_1 = ReactiveBaseline(l=self.Lambda)
        self.baseline_2 = ReactiveBaseline(l=self.Lambda)

    def set_random_seeds(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def calc_reinforce_loss(self, per_example_loss, which_agent_sequence, rewards, entropy_reg_coeff):
        loss = torch.stack(per_example_loss, dim=1)  # [B, T]
        mask = torch.tensor(which_agent_sequence).bool()
        mask = mask.repeat(loss.size(0), 1)
        not_mask = ~mask

        loss_1 = torch.masked_select(loss, not_mask).view(loss.size(0), -1)
        loss_2 = torch.masked_select(loss, mask).view(loss.size(0), -1)

        final_reward_1 = rewards["reward_1"] - \
            self.baseline_1.get_baseline_value()
        final_reward_2 = rewards["reward_2"] - \
            self.baseline_2.get_baseline_value()

        # Normalize rewards
        final_reward_1 = (final_reward_1 - final_reward_1.mean()
                          ) / (final_reward_1.std() + 1e-6)
        final_reward_2 = (final_reward_2 - final_reward_2.mean()
                          ) / (final_reward_2.std() + 1e-6)

        loss_1 = loss_1 * final_reward_1
        loss_2 = loss_2 * final_reward_2

        entropy_policy_1, entropy_policy_2 = self.entropy_reg_loss(
            per_example_loss)
        total_loss_1 = loss_1.mean() - entropy_reg_coeff * entropy_policy_1
        total_loss_2 = loss_2.mean() - entropy_reg_coeff * entropy_policy_2

        return total_loss_1, total_loss_2

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        mask = torch.tensor(self.which_agent_sequence).bool()
        not_mask = ~mask

        logits_1 = torch.masked_select(
            all_logits, not_mask).view(all_logits.size(0), -1)
        logits_2 = torch.masked_select(
            all_logits, mask).view(all_logits.size(0), -1)

        entropy_policy_1 = - \
            torch.mean(torch.sum(torch.exp(logits_1) * logits_1, dim=1))
        entropy_policy_2 = - \
            torch.mean(torch.sum(torch.exp(logits_2) * logits_2, dim=1))

        return entropy_policy_1, entropy_policy_2

    def train(self):
        self.batch_counter = 0
        debate_printer = Debate_Printer(
            self.output_dir, self.train_environment.grapher, self.num_rollouts
        )

        for episode in self.train_environment.get_episodes():
            is_train_judge = (self.batch_counter // self.train_judge_every) % 2 == 0 \
                or self.batch_counter >= self.rounds_sub_training

            which_agent_list = []
            logger.info(f"BATCH COUNTER: {self.batch_counter}")
            self.batch_counter += 1

            query_subjects = episode.get_query_subjects()
            query_relation = episode.get_query_relation()
            query_objects = episode.get_query_objects()
            episode_answers = episode.get_labels()

            self.judge.train() if is_train_judge else self.judge.eval()
            self.agent.train()

            logits = []
            rewards_1, rewards_2 = [], []

            for arguments in range(self.number_arguments):
                # Arguments for Agent 1
                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    which_agent_list.append(0.0)

                    next_relations = state['next_relations']
                    next_entities = state['next_entities']
                    current_entities = state['current_entities']

                    logits_1, rewards = self.agent.step(
                        next_relations, next_entities, current_entities, agent_id=1)
                    logits.append((0, rewards))
                    rewards_1.append(rewards)
                    state = episode(logits_1.argmax(dim=1))

                # Arguments for Agent 2
                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    which_agent_list.append(1.0)

                    next_relations = state['next_relations']
                    next_entities = state['next_entities']
                    current_entities = state['current_entities']

                    logits_2, rewards = self.agent.step(
                        next_relations, next_entities, current_entities, agent_id=2)
                    logits.append((1, rewards))
                    rewards_2.append(rewards)
                    state = episode(logits_2.argmax(dim=1))

            cum_discounted_reward_1, cum_discounted_reward_2 = self.calc_cum_discounted_reward(
                np.array(rewards_1), np.array(
                    rewards_2), np.array(which_agent_list)
            )

            if is_train_judge:
                loss_judge = self.judge.calc_loss(
                    query_subjects, query_relation, query_objects, episode_answers)
                self.optimizer_judge.zero_grad()
                loss_judge.backward()
                self.optimizer_judge.step()
            else:
                total_loss_1, total_loss_2 = self.calc_reinforce_loss(
                    logits, which_agent_list, {
                        "reward_1": cum_discounted_reward_1, "reward_2": cum_discounted_reward_2},
                    entropy_reg_coeff=self.decaying_beta
                )
                self.optimizer_agents.zero_grad()
                (total_loss_1 + total_loss_2).backward()
                self.optimizer_agents.step()

            if self.batch_counter % self.eval_every == 0:
                self.test(is_dev_environment=True)

    def test(self, is_dev_environment):
        self.agent.eval()
        self.judge.eval()

        environment = self.dev_test_environment if is_dev_environment else self.test_test_environment
        debate_printer = Debate_Printer(
            self.output_dir, self.train_environment.grapher, self.test_rollouts, is_append=True)

        total_examples = 0
        mean_reciprocal_rank = 0
        hits_at_1 = hits_at_3 = hits_at_10 = hits_at_20 = 0

        for episode in environment.get_episodes():
            temp_batch_size = episode.no_examples
            query_subjects = episode.get_query_subjects()
            query_relation = episode.get_query_relation()
            query_objects = episode.get_query_objects()
            episode_answers = episode.get_labels()

            previous_relation = torch.full((temp_batch_size * self.test_rollouts,),
                                           fill_value=self.relation_vocab['DUMMY_START_RELATION'], dtype=torch.long)
            agent_mem_1, agent_mem_2 = self.agent.get_init_state_array(
                temp_batch_size)

            debate_printer.create_debates(
                query_subjects, query_relation, query_objects, episode_answers)

            logits = []
            for arguments in range(self.number_arguments):
                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    next_relations = state['next_relations']
                    next_entities = state['next_entities']
                    current_entities = state['current_entities']

                    logits_1, agent_mem_1 = self.agent.step_test(
                        next_relations, next_entities, current_entities, previous_relation, agent_mem_1, agent_id=1
                    )
                    logits.append(logits_1)
                    state = episode(logits_1.argmax(dim=1))

                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    next_relations = state['next_relations']
                    next_entities = state['next_entities']
                    current_entities = state['current_entities']

                    logits_2, agent_mem_2 = self.agent.step_test(
                        next_relations, next_entities, current_entities, previous_relation, agent_mem_2, agent_id=2
                    )
                    logits.append(logits_2)
                    state = episode(logits_2.argmax(dim=1))

            # Compute metrics
            final_logits = torch.stack(logits).mean(dim=0)
            predictions = (final_logits > 0).int()
            accuracy = (predictions == torch.tensor(
                episode_answers)).float().mean()

            hits_at_1 += (predictions == 0).sum().item()
            hits_at_3 += (predictions < 3).sum().item()
            hits_at_10 += (predictions < 10).sum().item()
            hits_at_20 += (predictions < 20).sum().item()

            total_examples += temp_batch_size
            mean_reciprocal_rank += (1.0 /
                                     (predictions.nonzero().flatten() + 1)).sum().item()

        logger.info(f"Hits@1: {hits_at_1 / total_examples}")
        logger.info(f"Hits@3: {hits_at_3 / total_examples}")
        logger.info(f"Hits@10: {hits_at_10 / total_examples}")
        logger.info(f"Hits@20: {hits_at_20 / total_examples}")
        logger.info(f"MRR: {mean_reciprocal_rank / total_examples}")


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def main():
    """
    Runs an experiment or evaluates a pretrained model based on the value of the load_model option.

    If load_model is False, it trains models using a grid search over hyperparameter values. 
    If load_model is True, it evaluates a pretrained model.
    """
    option = read_options()

    # Set up logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = None

    # Read vocab files
    logger.info('Reading vocab files...')
    relation_vocab = json.load(
        open(option['vocab_dir'] + '/relation_vocab.json'))
    entity_vocab = json.load(open(option['vocab_dir'] + '/entity_vocab.json'))
    mid_to_name = (
        json.load(open(option['vocab_dir'] + '/fb15k_names.json'))
        if os.path.isfile(option['vocab_dir'] + '/fb15k_names.json')
        else None
    )
    logger.info('Total number of entities: {}'.format(len(entity_vocab)))
    logger.info('Total number of relations: {}'.format(len(relation_vocab)))

    if not option['load_model']:
        # Training phase
        for key, val in option.items():
            if not isinstance(val, list):
                option[key] = [val]

        best_permutation = None
        best_metric = 0

        for permutation in ParameterGrid(option):
            current_time = datetime.datetime.now().strftime('%y_%b_%d__%H_%M_%S')
            permutation['output_dir'] = os.path.join(
                permutation['base_output_dir'],
                f"{current_time}__{uuid.uuid4().hex[:4]}_{permutation['path_length']}_"
                f"{permutation['beta']}_{permutation['test_rollouts']}_{permutation['Lambda']}"
            )
            permutation['model_dir'] = os.path.join(
                permutation['output_dir'], 'model')
            permutation['relation_vocab'] = relation_vocab
            permutation['entity_vocab'] = entity_vocab
            permutation['mid_to_name'] = mid_to_name

            os.makedirs(permutation['output_dir'], exist_ok=True)
            os.mkdir(permutation['model_dir'])
            with open(os.path.join(permutation['output_dir'], 'config.txt'), 'w') as out:
                pprint(permutation, stream=out)

            logfile = logging.FileHandler(os.path.join(
                permutation['output_dir'], 'log.txt'), 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)

            trainer = Trainer(permutation, best_metric)

            # Train the model
            trainer.train()

            if trainer.best_metric > best_metric or best_permutation is None:
                best_metric = trainer.best_metric
                best_permutation = permutation

        # Test the best model
        current_time = datetime.datetime.now().strftime('%y_%b_%d__%H_%M_%S')
        best_permutation['output_dir'] = os.path.join(
            best_permutation['base_output_dir'],
            f"{current_time}__Test__{uuid.uuid4().hex[:4]}_{best_permutation['path_length']}_"
            f"{best_permutation['beta']}_{best_permutation['test_rollouts']}_{best_permutation['Lambda']}"
        )
        best_permutation['model_dir'] = os.path.join(
            best_permutation['output_dir'], 'model')
        best_permutation['relation_vocab'] = relation_vocab
        best_permutation['entity_vocab'] = entity_vocab
        best_permutation['mid_to_name'] = mid_to_name

        os.makedirs(best_permutation['output_dir'], exist_ok=True)
        os.mkdir(best_permutation['model_dir'])
        with open(os.path.join(best_permutation['output_dir'], 'config.txt'), 'w') as out:
            pprint(best_permutation, stream=out)

        trainer = Trainer(best_permutation, best_metric)
        trainer.test(is_dev_environment=False)

    else:
        # Load and evaluate pretrained model
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(option["model_load_dir"]))

        for key, value in option.items():
            if isinstance(value, list) and len(value) > 1:
                raise ValueError(
                    f"Parameter {key} has more than one value in the config file.")
            elif isinstance(value, list):
                option[key] = value[0]

        current_time = datetime.datetime.now().strftime('%y_%b_%d__%H_%M_%S')
        option['output_dir'] = os.path.join(
            option['base_output_dir'],
            f"{current_time}__Test__{uuid.uuid4().hex[:4]}_{option['path_length']}_"
            f"{option['beta']}_{option['test_rollouts']}_{option['Lambda']}"
        )
        option['model_dir'] = os.path.join(option['output_dir'], 'model')
        os.makedirs(option['output_dir'], exist_ok=True)
        os.mkdir(option['model_dir'])
        with open(os.path.join(option['output_dir'], 'config.txt'), 'w') as out:
            pprint(option, stream=out)

        logfile = logging.FileHandler(os.path.join(
            option['output_dir'], 'log.txt'), 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

        option['relation_vocab'] = relation_vocab
        option['entity_vocab'] = entity_vocab
        option['mid_to_name'] = mid_to_name

        trainer = Trainer(option, 0)

        # Evaluate the model
        trainer.test(is_dev_environment=True)
        if option['is_use_fixed_false_facts']:
            best_threshold = trainer.test(is_dev_environment=True)
            trainer.test(is_dev_environment=False,
                         best_threshold=best_threshold)


if __name__ == '__main__':
    main()

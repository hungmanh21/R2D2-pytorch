import datetime
import json
import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import random
from sklearn.model_selection import ParameterGrid
from pprint import pprint

# Assuming `agent.py` and `judge.py` files have been modified for compatibility
from code.model.judge import Judge
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
from code.model.baseline import ReactiveBaseline
from code.model.debate_printer import Debate_Printer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class Trainer:
    def __init__(self, params, best_metric):
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

        # Optimizers for judge and agents
        self.learning_rate_judge = params['learning_rate_judge']
        self.optimizer_judge = optim.Adam(
            self.judge.parameters(), lr=self.learning_rate_judge)
        self.optimizer_agents = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate_agents)

        # Baseline models for REINFORCE loss
        self.baseline_1 = ReactiveBaseline(l=self.Lambda)
        self.baseline_2 = ReactiveBaseline(l=self.Lambda)

    def set_random_seeds(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def calc_reinforce_loss(self, per_example_loss, which_agent_sequence, rewards, entropy_reg_coeff):
        loss = torch.stack(per_example_loss, dim=1)
        mask = torch.tensor(which_agent_sequence).bool()
        mask = mask.repeat(loss.size(0), 1)
        not_mask = ~mask

        loss_1 = torch.masked_select(loss, not_mask).view(loss.size(0), -1)
        loss_2 = torch.masked_select(loss, mask).view(loss.size(0), -1)

        final_reward_1 = rewards["reward_1"] - \
            self.baseline_1.get_baseline_value()
        final_reward_2 = rewards["reward_2"] - \
            self.baseline_2.get_baseline_value()

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
        all_logits = torch.stack(all_logits, dim=2)
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
        self.agent.train()
        self.judge.train()
        self.batch_counter = 0
        debate_printer = Debate_Printer(
            self.output_dir, self.train_environment.grapher, self.num_rollouts)

        for episode in self.train_environment.get_episodes():
            is_train_judge = (self.batch_counter // self.train_judge_every) % 2 == 0 \
                or self.batch_counter >= self.rounds_sub_training

            self.batch_counter += 1

            query_subjects = episode.get_query_subjects()
            query_relation = episode.get_query_relation()
            query_objects = episode.get_query_objects()
            labels = episode.get_labels()

            # Set query embeddings
            self.agent.set_query_embeddings(
                query_subjects, query_relation, query_objects)
            self.judge.set_query_embeddings(
                query_subjects, query_relation, query_objects)

            # Run agent
            loss_judge, final_logit_judge, _, per_example_loss, per_example_logits, action_idx, rewards_agents, _ = \
                self.agent(
                    which_agent=[0.0] * (self.number_steps // 2) +
                    [1.0] * (self.number_steps // 2),
                    candidate_relation_sequence=episode.state["next_relations"],
                    candidate_entity_sequence=episode.state["next_entities"],
                    current_entities=episode.state["current_entities"],
                    range_arr=torch.arange(
                        self.batch_size, device=self.agent.device),
                    T=self.number_steps,
                    random_flag=False,
                )

            if is_train_judge:
                self.optimizer_judge.zero_grad()
                loss_judge.backward()
                self.optimizer_judge.step()
            else:
                # Compute REINFORCE loss for agents
                loss_1, loss_2 = self.calc_reinforce_loss(
                    per_example_loss,
                    which_agent_sequence=[
                        0.0] * (self.number_steps // 2) + [1.0] * (self.number_steps // 2),
                    rewards={
                        "reward_1": rewards_agents[:self.number_steps // 2],
                        "reward_2": rewards_agents[self.number_steps // 2:],
                    },
                    entropy_reg_coeff=self.decaying_beta,
                )
                self.optimizer_agents.zero_grad()
                (loss_1 + loss_2).backward()
                self.optimizer_agents.step()

            if self.batch_counter % self.eval_every == 0:
                self.test(is_dev_environment=True)

    def test(self, is_dev_environment):
        self.agent.eval()
        self.judge.eval()
        environment = self.dev_test_environment if is_dev_environment else self.test_test_environment
        debate_printer = Debate_Printer(
            self.output_dir, self.train_environment.grapher, self.test_rollouts, is_append=True)

        hits_at_1, hits_at_3, hits_at_10, hits_at_20 = 0, 0, 0, 0
        mean_reciprocal_rank, total_examples = 0, 0

        for episode in environment.get_episodes():
            query_subjects = episode.get_query_subjects()
            query_relation = episode.get_query_relation()
            query_objects = episode.get_query_objects()
            labels = episode.get_labels()

            self.agent.set_query_embeddings(
                query_subjects, query_relation, query_objects)
            self.judge.set_query_embeddings(
                query_subjects, query_relation, query_objects)

            logits, _, _, _, _, action_idx, _, _ = self.agent(
                which_agent=[0.0] * (self.number_steps // 2) +
                [1.0] * (self.number_steps // 2),
                candidate_relation_sequence=episode.state["next_relations"],
                candidate_entity_sequence=episode.state["next_entities"],
                current_entities=episode.state["current_entities"],
                range_arr=torch.arange(
                    episode.no_examples, device=self.agent.device),
                T=self.number_steps,
                random_flag=False,
            )

            logits = logits.detach()
            predictions = torch.sigmoid(logits).squeeze().round()

            hits_at_1 += (predictions == labels).sum().item()
            total_examples += len(labels)

            ranks = torch.argsort(torch.argsort(-logits, dim=0), dim=0) + 1
            mean_reciprocal_rank += (1 / ranks).sum().item()
            hits_at_3 += (ranks <= 3).sum().item()
            hits_at_10 += (ranks <= 10).sum().item()
            hits_at_20 += (ranks <= 20).sum().item()

        logger.info(f"Hits@1: {hits_at_1 / total_examples:.4f}")
        logger.info(f"Hits@3: {hits_at_3 / total_examples:.4f}")
        logger.info(f"Hits@10: {hits_at_10 / total_examples:.4f}")
        logger.info(f"Hits@20: {hits_at_20 / total_examples:.4f}")
        logger.info(f"MRR: {mean_reciprocal_rank / total_examples:.4f}")


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

"""
PyTorch implementation of the R2D2 Trainer
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
import gc
import datetime
import uuid
from pprint import pprint
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_curve
from tqdm import tqdm

from code.model.agent import Agent
from code.model.debate_printer import Debate_Printer
from code.model.environment import Environment
from code.model.judge import Judge
from code.options import read_options
import torch.nn.functional as F

logger = logging.getLogger()


class ReactiveBaseline:
    """Simple moving average baseline."""

    def __init__(self, l: float):
        self.l = l
        self.value = 0.0

    def update(self, new_value: float):
        self.value = self.l * self.value + (1 - self.l) * new_value

    def get_baseline_value(self) -> float:
        return self.value


class Trainer:
    """
    PyTorch implementation of the R2D2 Trainer.
    Coordinates training and evaluation of the debate system.
    """

    def __init__(self, params: Dict, best_metric: float):
        """
        Initialize trainer with parameters and best metric.

        Args:
            params: Dictionary of training parameters
            best_metric: Best metric achieved so far
        """
        # Transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        # Set random seeds if provided
        if self.seed is not None:
            print("SEED :", self.seed)
            torch.manual_seed(self.seed[0])
            np.random.seed(self.seed[0])

        self.batch_size = self.batch_size * \
            (1 + self.false_facts_train) * self.num_rollouts

        # Initialize models
        self.judge = Judge(params).to(self.device)
        self.agent = Agent(params, self.judge).to(self.device)

        # Initialize environments
        self.train_environment = Environment(params, 'train')
        self.dev_test_environment = Environment(params, 'dev')
        self.test_test_environment = Environment(params, 'test')

        self.number_steps = self.path_length * self.number_arguments * 2
        self.best_metric = best_metric

        # Setup optimizers
        self.learning_rate_judge_init = params['learning_rate_judge']
        self.learning_rate_judge = self.learning_rate_judge_init

        self.optimizer_judge = optim.Adam(
            self.judge.parameters(), lr=self.learning_rate_judge)
        self.optimizer_agents = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate_agents)

        # Setup baselines
        self.baseline_1 = ReactiveBaseline(l=self.Lambda)
        self.baseline_2 = ReactiveBaseline(l=self.Lambda)

    def calc_reinforce_loss(self, per_example_loss: List[torch.Tensor],
                            which_agent_sequence: List[torch.Tensor],
                            per_example_logits: List[torch.Tensor],
                            cum_discounted_reward_1: torch.Tensor,
                            cum_discounted_reward_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate REINFORCE loss for both agents.
        """
        # Stack losses and create mask
        loss = torch.stack(per_example_loss, dim=1)
        mask = torch.stack([x == 0 for x in which_agent_sequence], dim=1)
        not_mask = ~mask

        # Split losses by agent
        loss_1 = loss[not_mask].view(loss.size(0), -1)
        loss_2 = loss[mask].view(loss.size(0), -1)

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
        reward_mean_1 = final_reward_1.mean()
        reward_std_1 = final_reward_1.std() + 1e-6
        reward_mean_2 = final_reward_2.mean()
        reward_std_2 = final_reward_2.std() + 1e-6

        final_reward_1 = (final_reward_1 - reward_mean_1) / reward_std_1
        final_reward_2 = (final_reward_2 - reward_mean_2) / reward_std_2

        # Calculate losses with rewards
        loss_1 = (loss_1 * final_reward_1).mean()
        loss_2 = (loss_2 * final_reward_2).mean()

        # Add entropy regularization
        entropy_1, entropy_2 = self.entropy_reg_loss(per_example_logits)
        total_loss_1 = loss_1 - self.beta * entropy_1
        total_loss_2 = loss_2 - self.beta * entropy_2

        return total_loss_1, total_loss_2

    def entropy_reg_loss(self, all_logits: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate entropy regularization loss for both agents.

        Args:
            all_logits: List of logits for each step

        Returns:
            Tuple of entropy values for both agents
        """
        all_logits = torch.stack(all_logits, dim=2)
        agent_1_mask = torch.tensor([x == 0 for x in range(
            all_logits.size(2))], device=all_logits.device)
        agent_2_mask = ~agent_1_mask

        # Get logits for each agent
        logits_1 = all_logits[:, :, agent_1_mask]
        logits_2 = all_logits[:, :, agent_2_mask]

        # Calculate entropy
        probs_1 = torch.softmax(logits_1, dim=1)
        probs_2 = torch.softmax(logits_2, dim=1)

        entropy_1 = -(probs_1 * logits_1).sum(dim=1).mean()
        entropy_2 = -(probs_2 * logits_2).sum(dim=1).mean()

        return entropy_1, entropy_2

    def train(self, sess):
        """
        Train the model.
        """
        self.batch_counter = 0
        debate_printer = Debate_Printer(
            self.output_dir, self.train_environment.grapher, self.num_rollouts)

        for episode in self.train_environment.get_episodes():
            is_train_judge = (self.batch_counter // self.train_judge_every) % 2 == 0 \
                or self.batch_counter >= self.rounds_sub_training

            logger.info(f"BATCH COUNTER: {self.batch_counter}")
            self.batch_counter += 1

            # Move episode data to device
            query_subjects = episode.get_query_subjects().to(self.device)
            query_relations = episode.get_query_relation().to(self.device)
            query_objects = episode.get_query_objects().to(self.device)
            episode_labels = episode.get_labels().to(self.device)

            # Reset gradients
            self.optimizer_judge.zero_grad() if is_train_judge else self.optimizer_agents.zero_grad()

            # Initialize tracking variables
            which_agent_list = []
            all_loss = []
            all_logits = []
            action_indices = []
            all_rewards = []
            all_rewards_before_baseline = []

            debate_printer.create_debates(
                query_subjects.cpu().numpy(),
                query_relations.cpu().numpy(),
                query_objects.cpu().numpy(),
                episode_labels.cpu().numpy()
            )

            # Execute debate steps
            for argument in range(self.number_arguments):
                # Pro agent turn
                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    which_agent_list.append(0.0)

                    # Convert state tensors to device
                    next_relations = torch.from_numpy(
                        state['next_relations']).to(self.device)
                    next_entities = torch.from_numpy(
                        state['next_entities']).to(self.device)
                    current_entities = torch.from_numpy(
                        state['current_entities']).to(self.device)

                    # Execute step
                    loss, pro_memory, con_memory, logits, action_idx, rewards, print_rewards = self.agent.step(
                        next_relations,
                        next_entities,
                        pro_memory,
                        con_memory,
                        current_entities,
                        which_agent=0.0,
                        is_random=self.batch_counter < self.train_judge_every
                    )

                    # Track results
                    all_loss.append(loss)
                    all_logits.append(logits)
                    action_indices.append(action_idx)
                    all_rewards.append(rewards)
                    all_rewards_before_baseline.append(print_rewards)

                    # Update state
                    state = episode(action_idx.cpu().numpy())

                    # Print debate progress
                    rel_string, ent_string = debate_printer.get_action_rel_ent(
                        action_idx.cpu().numpy(),
                        state
                    )
                    debate_printer.create_arguments(
                        [rel_string], [ent_string],
                        rewards.cpu().numpy(),
                        True
                    )

                # Con agent turn
                state = episode.reset_initial_state()
                for path_num in range(self.path_length):
                    which_agent_list.append(1.0)

                    # Similar process for con agent...
                    next_relations = torch.from_numpy(
                        state['next_relations']).to(self.device)
                    next_entities = torch.from_numpy(
                        state['next_entities']).to(self.device)
                    current_entities = torch.from_numpy(
                        state['current_entities']).to(self.device)

                    loss, pro_memory, con_memory, logits, action_idx, rewards, print_rewards = self.agent.step(
                        next_relations,
                        next_entities,
                        pro_memory,
                        con_memory,
                        current_entities,
                        which_agent=1.0,
                        is_random=self.batch_counter < self.train_judge_every
                    )

                    all_loss.append(loss)
                    all_logits.append(logits)
                    action_indices.append(action_idx)
                    all_rewards.append(rewards)
                    all_rewards_before_baseline.append(print_rewards)

                    state = episode(action_idx.cpu().numpy())

                    rel_string, ent_string = debate_printer.get_action_rel_ent(
                        action_idx.cpu().numpy(),
                        state
                    )
                    debate_printer.create_arguments(
                        [rel_string], [ent_string],
                        rewards.cpu().numpy(),
                        False
                    )
            final_logits = self.judge(all_rewards)
            debate_printer.set_debates_final_logit(final_logits.cpu().numpy())

            # Calculate accuracy
            predictions = (final_logits > 0).float()
            accuracy = (predictions == episode_labels).float().mean()

            logger.info(f"Mean label: {episode_labels.float().mean()}")
            logger.info(f"Accuracy: {accuracy.item()}")

            # Train judge or agents
            if is_train_judge:
                judge_loss = F.binary_cross_entropy_with_logits(
                    final_logits, episode_labels.float())
                judge_loss.backward()
                self.optimizer_judge.step()

                logger.info(f"Judge Loss: {judge_loss.item()}")
            else:
                # Calculate rewards and REINFORCE loss
                rewards_1, rewards_2 = episode.get_rewards(
                    list(zip(which_agent_list, all_rewards)))

                cum_discounted_reward_1, cum_discounted_reward_2 = self.calc_cum_discounted_reward(
                    rewards_1.to(self.device),
                    rewards_2.to(self.device),
                    torch.tensor(which_agent_list, device=self.device)
                )

                # Update baselines
                if not self.custom_baseline:
                    self.baseline_1.update(
                        cum_discounted_reward_1.mean().item())
                    self.baseline_2.update(
                        cum_discounted_reward_2.mean().item())

                # Calculate agent losses
                loss_1, loss_2 = self.calc_reinforce_loss(
                    all_loss,
                    which_agent_list,
                    all_logits,
                    cum_discounted_reward_1,
                    cum_discounted_reward_2
                )

                total_loss = loss_1 + loss_2
                total_loss.backward()
                self.optimizer_agents.step()

                logger.info(f"Agent 1 Loss: {loss_1.item()}")
                logger.info(f"Agent 2 Loss: {loss_2.item()}")

            # Save model and evaluate periodically
            if self.batch_counter == self.rounds_sub_training:
                self.save_model('unbiased_model')

            if self.batch_counter % self.eval_every == 0:
                self.test(True)  # Test on dev set

            # Memory management
            if os.name == 'posix':
                logger.info(
                    f'Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KB')

            gc.collect()
            torch.cuda.empty_cache()

            if self.batch_counter >= self.total_iterations:
                break

    def test(self, is_dev_environment: bool, save_model: bool = False, best_threshold: Optional[float] = None):
        """
        Test the model on dev or test set.

        Args:
            is_dev_environment: Whether to use dev or test environment
            save_model: Whether to save the model
            best_threshold: Optional threshold for binary classification
        """
        self.agent.eval()
        self.judge.eval()

        batch_counter = 0
        total_examples = 0
        all_probs = []
        all_labels = []
        logits_list = []
        correct_answer_list = []

        environment = self.dev_test_environment if is_dev_environment else self.test_test_environment
        debate_printer = Debate_Printer(
            self.output_dir, environment.grapher, self.test_rollouts, is_append=True)

        with torch.no_grad():
            for episode in tqdm(environment.get_episodes()):
                batch_counter += 1
                temp_batch_size = episode.no_examples

                # Process episode similar to training
                query_subjects = episode.get_query_subjects().to(self.device)
                query_relations = episode.get_query_relation().to(self.device)
                query_objects = episode.get_query_objects().to(self.device)
                episode_labels = episode.get_labels().to(self.device)

                # Initialize tracking variables
                pro_memory = self.agent.get_init_state_array(temp_batch_size)[
                    0].to(self.device)
                con_memory = self.agent.get_init_state_array(temp_batch_size)[
                    1].to(self.device)

                debate_printer.create_debates(
                    query_subjects.cpu().numpy(),
                    query_relations.cpu().numpy(),
                    query_objects.cpu().numpy(),
                    episode_labels.cpu().numpy()
                )

                # Execute debate steps
                for argument in range(self.number_arguments):
                    # Similar structure to training loop but without backward pass
                    state = episode.reset_initial_state()

                    # Pro agent turn
                    for path_num in range(self.path_length):
                        next_relations = torch.from_numpy(
                            state['next_relations']).to(self.device)
                        next_entities = torch.from_numpy(
                            state['next_entities']).to(self.device)
                        current_entities = torch.from_numpy(
                            state['current_entities']).to(self.device)

                        _, pro_memory, con_memory, _, action_idx, rewards, _ = self.agent.step(
                            next_relations,
                            next_entities,
                            pro_memory,
                            con_memory,
                            current_entities,
                            which_agent=0.0,
                            is_random=False
                        )

                        state = episode(action_idx.cpu().numpy())

                        rel_string, ent_string = debate_printer.get_action_rel_ent(
                            action_idx.cpu().numpy(),
                            state
                        )
                        debate_printer.create_arguments(
                            [rel_string], [ent_string],
                            rewards.cpu().numpy(),
                            True
                        )

                    # Con agent turn
                    state = episode.reset_initial_state()
                    for path_num in range(self.path_length):
                        # Similar process for con agent...
                        next_relations = torch.from_numpy(
                            state['next_relations']).to(self.device)
                        next_entities = torch.from_numpy(
                            state['next_entities']).to(self.device)
                        current_entities = torch.from_numpy(
                            state['current_entities']).to(self.device)

                        _, pro_memory, con_memory, _, action_idx, rewards, _ = self.agent.step(
                            next_relations,
                            next_entities,
                            pro_memory,
                            con_memory,
                            current_entities,
                            which_agent=1.0,
                            is_random=False
                        )

                        state = episode(action_idx.cpu().numpy())

                        rel_string, ent_string = debate_printer.get_action_rel_ent(
                            action_idx.cpu().numpy(),
                            state
                        )
                        debate_printer.create_arguments(
                            [rel_string], [ent_string],
                            rewards.cpu().numpy(),
                            False
                        )

                # Get final prediction
                final_logits = self.judge(rewards).reshape(temp_batch_size, -1)
                debate_printer.set_debates_final_logit(
                    final_logits.cpu().numpy())

                # Track metrics
                logits_list.append(final_logits)
                correct_answer_list.append(episode_labels)
                all_probs.extend(torch.sigmoid(final_logits).cpu().numpy())
                all_labels.extend(episode_labels.cpu().numpy())

                total_examples += temp_batch_size

        # Calculate metrics
        all_logits = torch.cat(logits_list).cpu().numpy()
        all_labels = np.array(all_labels)

        # Calculate PR and ROC curves
        precision, recall, thresholds = precision_recall_curve(
            all_labels, all_probs)
        auc_pr = auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        auc_roc = auc(fpr, tpr)

        # Find best threshold if not provided
        if best_threshold is None:
            best_acc = 0
            best_threshold = 0
            for threshold in thresholds:

                preds = (all_probs > threshold).astype(int)
                acc = accuracy_score(all_labels, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
        else:
            preds = (all_probs > best_threshold).astype(int)
            best_acc = accuracy_score(all_labels, preds)

        # Log metrics
        logger.info("========== METRICS =============")
        logger.info(f"Best Threshold: {best_threshold}")
        logger.info(f"Accuracy: {best_acc}")
        logger.info(f"AUC-PR: {auc_pr}")
        logger.info(f"AUC-ROC: {auc_roc}")
        logger.info("===============================")

        # Save model if improved
        if self.is_use_fixed_false_facts:
            if best_acc > self.best_metric:
                self.save_model("model")
            self.best_metric = max(best_acc, self.best_metric)
            self.best_threshold = best_threshold
        else:
            if auc_pr > self.best_metric:
                self.save_model("model")
            self.best_metric = max(auc_pr, self.best_metric)

        # Reset to training mode
        self.agent.train()
        self.judge.train()

        return best_threshold

    def save_model(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.model_dir, f"{name}.pt")
        torch.save({
            'judge_state_dict': self.judge.state_dict(),
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_judge_state_dict': self.optimizer_judge.state_dict(),
            'optimizer_agents_state_dict': self.optimizer_agents.state_dict(),
            'best_metric': self.best_metric,
            'best_threshold': self.best_threshold if hasattr(self, 'best_threshold') else None
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.judge.load_state_dict(checkpoint['judge_state_dict'])
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer_judge.load_state_dict(
            checkpoint['optimizer_judge_state_dict'])
        self.optimizer_agents.load_state_dict(
            checkpoint['optimizer_agents_state_dict'])
        self.best_metric = checkpoint['best_metric']
        if checkpoint['best_threshold'] is not None:
            self.best_threshold = checkpoint['best_threshold']
        logger.info(f"Model loaded from {path}")

    def calc_cum_discounted_reward(self, rewards_1: torch.Tensor, rewards_2: torch.Tensor,
                                   which_agent_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cumulative discounted rewards for both agents.

        Args:
            rewards_1: Rewards for agent 1
            rewards_2: Rewards for agent 2
            which_agent_sequence: Sequence indicating which agent took each action

        Returns:
            Tuple of cumulative discounted rewards for both agents
        """
        # Count actions per agent
        num_actions_1 = (which_agent_sequence == 0).sum().item()
        num_actions_2 = (which_agent_sequence == 1).sum().item()

        # Initialize reward tensors
        cum_disc_reward_1 = torch.zeros(
            rewards_1.shape[0], num_actions_1, device=rewards_1.device)
        cum_disc_reward_2 = torch.zeros(
            rewards_2.shape[0], num_actions_2, device=rewards_2.device)

        r_1_index = -1
        r_2_index = -1
        running_add_1 = torch.zeros(
            rewards_1.shape[0], device=rewards_1.device)
        running_add_2 = torch.zeros(
            rewards_2.shape[0], device=rewards_2.device)
        prev_agent = None

        # Calculate discounted rewards in reverse
        for t in reversed(range(len(which_agent_sequence))):
            current_agent = which_agent_sequence[t].item()

            if current_agent == 0:  # Agent 1
                if prev_agent == 0:
                    running_add_1 = self.gamma * \
                        running_add_1 + rewards_1[:, r_1_index]
                else:
                    running_add_1 = rewards_1[:, r_1_index]
                cum_disc_reward_1[:, r_1_index] = running_add_1
                r_1_index -= 1
            else:  # Agent 2
                if prev_agent == 1:
                    running_add_2 = self.gamma * \
                        running_add_2 + rewards_2[:, r_2_index]
                else:
                    running_add_2 = rewards_2[:, r_2_index]
                cum_disc_reward_2[:, r_2_index] = running_add_2
                r_2_index -= 1

            prev_agent = current_agent

        return cum_disc_reward_1, cum_disc_reward_2


def main():
    """Main training/testing function."""
    options = read_options()

    # Setup logging
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Load vocabularies
    logger.info('Reading vocabulary files...')
    with open(os.path.join(options['vocab_dir'], 'relation_vocab.json'), 'r') as f:
        relation_vocab = json.load(f)
    with open(os.path.join(options['vocab_dir'], 'entity_vocab.json'), 'r') as f:
        entity_vocab = json.load(f)

    mid_to_name_path = os.path.join(options['vocab_dir'], 'fb15k_names.json')
    mid_to_name = json.load(open(mid_to_name_path)) if os.path.exists(
        mid_to_name_path) else None

    logger.info(f'Total number of entities: {len(entity_vocab)}')
    logger.info(f'Total number of relations: {len(relation_vocab)}')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not options['no_cuda'] else 'cpu')
    logger.info(f'Using device: {device}')

    if not options['load_model']:
        # Training mode
        best_metric = 0
        best_model = None

        # Create output directory
        timestamp = datetime.datetime.now().strftime('%y_%b_%d__%H_%M_%S')
        model_id = str(uuid.uuid4())[:4]
        output_dir = os.path.join(
            options['base_output_dir'],
            f"{timestamp}__{model_id}_{options['path_length']}_{options['beta']}_{options['test_rollouts']}_{options['Lambda']}"
        )
        model_dir = os.path.join(output_dir, 'model')
        os.makedirs(output_dir)
        os.makedirs(model_dir)

        # Save configuration
        options['output_dir'] = output_dir
        options['model_dir'] = model_dir
        options['relation_vocab'] = relation_vocab
        options['entity_vocab'] = entity_vocab
        options['mid_to_name'] = mid_to_name

        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(options, f, indent=4)

        # Initialize trainer
        for key, value in options.items():
            if isinstance(value, list):
                options[key] = value[0]
        trainer = Trainer(options, best_metric)
        trainer.to(device)

        # Train model
        trainer.train()

        # Test best model
        if options['is_use_fixed_false_facts']:
            trainer.test(False, True, trainer.best_threshold)
        else:
            trainer.test(False, True)
    else:
        # Testing mode
        logger.info(f"Loading model from {options['model_load_dir']}")

        # Create output directory for test results
        timestamp = datetime.datetime.now().strftime('%y_%b_%d__%H_%M_%S')
        output_dir = os.path.join(
            options['base_output_dir'],
            f"{timestamp}__Test__{options['path_length']}_{options['beta']}_{options['test_rollouts']}_{options['Lambda']}"
        )
        model_dir = os.path.join(output_dir, 'model')
        os.makedirs(output_dir)
        os.makedirs(model_dir)

        # Save test configuration
        options['output_dir'] = output_dir
        options['model_dir'] = model_dir
        options['relation_vocab'] = relation_vocab
        options['entity_vocab'] = entity_vocab
        options['mid_to_name'] = mid_to_name

        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(options, f, indent=4)

        # Initialize trainer and load model
        trainer = Trainer(options, 0)
        trainer.to(device)
        trainer.load_model(os.path.join(options['model_load_dir'], 'model.pt'))

        # Run tests
        if options['is_use_fixed_false_facts']:
            best_threshold = trainer.test(True, False)  # Dev set
            trainer.test(False, True, best_threshold)  # Test set
        else:
            trainer.test(False, True)  # Test set


if __name__ == '__main__':
    main()

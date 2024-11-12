from code.model.environment import env
from code.options import read_options
import logging
import json
import datetime
import os
import uuid
from pprint import pprint
from code.model.trainer import create_permutations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


logger = logging.getLogger()

class SimpleClassifier(nn.Module):
    """
    Simple classifier for fact prediction consisting of a single dense layer. 
    Uses the query relation and object for prediction.
    """

    def __init__(self, params):
        super(SimpleClassifier, self).__init__()
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.train_env = env(params,'train')
        self.test_env = env(params,'test')
        self.eval_every = params['eval_every']
        self.learning_rate = params['learning_rate_judge']
        self.total_iteration = params['total_iterations']

        self.relation_embeddings = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size)
        self.entity_embeddings = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size)
        self.classifier = nn.Linear(4 * self.embedding_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, query_relation, query_object):
        relation_embedding = self.relation_embeddings(query_relation)
        object_embedding = self.entity_embeddings(query_object)
        query_embedding = torch.cat([relation_embedding, object_embedding], dim=-1)
        logits = self.classifier(query_embedding)
        return logits

    def train(self, device):
        counter = 0
        for episode in self.train_env.get_episodes():
            query_relation = episode.get_query_relation().to(device)
            query_object = episode.get_query_objects().to(device)
            label = episode.get_labels().to(device)

            self.optimizer.zero_grad()
            logits = self.forward(query_relation, query_object)
            loss = nn.BCEWithLogitsLoss()(logits, label)
            loss.backward()
            self.optimizer.step()

            predictions = (logits > 0).float()
            acc = torch.mean((predictions == label).float())
            print(acc.item())

            counter += 1
            if counter >= self.total_iteration:
                break

        self.eval(device)

    def eval(self, device):
        total_acc = 0
        total_examples = 0
        for episode in self.test_env.get_episodes():
            query_relation = episode.get_query_relation().to(device)
            query_object = episode.get_query_objects().to(device)
            label = episode.get_labels().to(device)
            temp_batch_size = episode.no_examples

            logits = self.forward(query_relation, query_object)
            predictions = (logits > 0).float()

            total_acc += torch.sum(predictions == label)
            total_examples += temp_batch_size

        logger.info("Acc_Test === {}".format(total_acc / total_examples))

def main():
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = None
    logger.addHandler(console)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    relation_vocab = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    entity_vocab = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    logger.info('Reading mid to name map')
    mid_to_word = {}
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))
    save_path = ''

    best_permutation = None
    best_acc = 0
    for permutation in create_permutations(options):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%y_%b_%d__%H_%M_%S')
        permutation['output_dir'] = options['base_output_dir'] + '/' + str(current_time) + '__' + str(uuid.uuid4())[
                                                                                                  :4] + '_' + str(
            permutation['path_length']) + '_' + str(permutation['beta']) + '_' + str(
            permutation['test_rollouts']) + '_' + str(
            permutation['Lambda'])

        permutation['model_dir'] = permutation['output_dir'] + '/' + 'model/'

        permutation['load_model'] = (permutation['load_model'] == 1)

        ##Logger##
        permutation['path_logger_file'] = permutation['output_dir']
        permutation['log_file_name'] = permutation['output_dir'] + '/log.txt'
        os.makedirs(permutation['output_dir'])
        os.mkdir(permutation['model_dir'])
        with open(permutation['output_dir'] + '/config.txt', 'w') as out:
            pprint(permutation, stream=out)

        # print and return
        maxLen = max([len(ii) for ii in permutation.keys()])
        fmtString = '\t%' + str(maxLen) + 's : %s'
        print('Arguments:')
        for keyPair in sorted(permutation.items()): print(fmtString % keyPair)
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(permutation['log_file_name'], 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        permutation['relation_vocab'] = relation_vocab
        permutation['entity_vocab'] = entity_vocab

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleClassifier(permutation).to(device)
        model.train(device)

if __name__ == "__main__":
    main()
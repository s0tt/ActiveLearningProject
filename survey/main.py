# delay evaluation of annotation
from __future__ import annotations

import glob
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from functools import reduce
from operator import attrgetter
from typing import Set, Union

import numpy
from tensorboard.compat.proto.event_pb2 import _EVENT
import torch
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedTokenizer

from agent import MRQAAgent
from data import (Dataset, MRQADataset, SharedTaskDatasetReader,
                  SlidingWindowHandler)
from utils import (GLOBAL_SEED, allocate_mem, init_random, store_metrics,
                   update_dict)

reader = SharedTaskDatasetReader(answer_first_occurence_only = True)

DATASETS = [
Dataset('SQuAD-train', 'train/SQuAD.jsonl.gz', reader),
Dataset('SQuAD-dev', 'dev/SQuAD.jsonl.gz', reader),
Dataset('HotpotQA-train', 'train/HotpotQA.jsonl.gz', reader),
Dataset('HotpotQA-dev', 'dev/HotpotQA.jsonl.gz', reader),
Dataset('TriviaQA-train', 'train/TriviaQA.jsonl.gz', reader),
Dataset('TriviaQA-dev', 'dev/TriviaQA.jsonl.gz', reader),
Dataset('NewsQA-train', 'train/NewsQA.jsonl.gz', reader),
Dataset('NewsQA-dev', 'dev/NewsQA.jsonl.gz', reader),
Dataset('SearchQA-train', 'train/SearchQA.jsonl.gz', reader),
Dataset('SearchQA-dev', 'dev/SearchQA.jsonl.gz', reader),
Dataset('NaturalQuestionsShort-train', 'train/NaturalQuestions.jsonl.gz', reader),
Dataset('NaturalQuestionsShort-dev', 'dev/NaturalQuestions.jsonl.gz', reader),
Dataset('DROP-dev', 'dev/DROP.jsonl.gz', reader),
Dataset('RACE-dev', 'dev/RACE.jsonl.gz', reader),
Dataset('BioASQ-dev', 'dev/BioASQ.jsonl.gz', reader),
Dataset('TextbookQA-dev', 'dev/TextbookQA.jsonl.gz', reader),
Dataset('RelationExtraction-dev', 'dev/RelationExtraction.jsonl.gz', reader),
Dataset('DuoRC-dev', 'dev/DuoRC.jsonl.gz', reader),
]


def check_positive(string):
    try:
        value = int(string)
    except:
        msg = "invalid value: %r (has to be a number)" % string
        raise argparse.ArgumentTypeError(msg)
    if value <= 0:
        msg = "invalid value: %r (has to be larger than 0)" % string
        raise argparse.ArgumentTypeError(msg)
    return value


def expand_path(path: str):
    return os.path.abspath(os.path.expanduser(path))


def match_datasets(data_dir: str, search: Union[None, str], no_checks: bool = False, eval_samples: int = None):
    if search is None:
        return []
    
    datasets = set()
    for given in search:
        for dataset in DATASETS:
            if given == 'all':
                datasets.add(dataset)
            elif re.match(given.split('/')[0], dataset.name, re.IGNORECASE):
                split = given.split('/')
                num_samples = -1
                num_samples_eval = eval_samples
                samples_total = 0
                if len(split) > 1:
                    # there is a number of samples given
                    try:
                        num_samples = int(split[1])
                        samples_total += num_samples
                    except:
                        pass
                if len(split) > 2:
                    # there is a number of samples for evaluation given
                    try:
                        num_samples_eval = int(split[2])
                    except:
                        pass
                samples_total += num_samples_eval if num_samples_eval is not None else 0
                if len(split) > 1:
                    # check whether requested samples exceed total samples
                    dataset_num_samples = dataset.get_total_samples(data_dir)
                    if samples_total > dataset_num_samples:
                        logging.error("Skipping dataset %s since amount of total samples (%d) exceeds the number of samples (%d) in the dataset!", dataset.name, samples_total, dataset_num_samples)
                        continue

                datasets.add(replace(dataset, num_samples=num_samples, num_samples_eval=num_samples_eval))
    dataset_names_duplicates = [name for name, count in Counter(map(attrgetter('name'), datasets)).items() if count > 1]
    if not no_checks:
        # check if datset occurs multiple times (with a different amount of samples)
        if dataset_names_duplicates:
            logging.warning("The following datasets appear multiple times (with a different amount of samples): %s", ''.join(dataset_names_duplicates))
    return datasets


def get_datasets(data_dir: str, cache_dir: str, sample_processor: callable, tokenizer: PreTrainedTokenizer, datasets: Set[Dataset], seed: int = None, force_preprocess: bool = False):
    data, data_split = [], []

    for dataset in datasets:
        _data = MRQADataset.load(data_dir, cache_dir, dataset, sample_processor, tokenizer, force_preprocess=force_preprocess)
        num_samples_eval = dataset.num_samples_eval
        if num_samples_eval is not None:
            logging.info("Drawing %d sample(s) with seed %d from %s for evaluation", num_samples_eval, args.eval_seed, _data)
            rng = numpy.random.RandomState(args.eval_seed)
            data_split.append(_data.sample_split(num_samples_eval, shuffle_fn=rng.shuffle, remove_samples=True, seed=args.eval_seed))
        if dataset.num_samples > 0:
            logging.info("Drawing %d sample(s) uniformly at random from %s", dataset.num_samples, _data)
            data.append(_data.sample_split(dataset.num_samples, seed=seed))
        elif dataset.num_samples == -1:
            data.append(_data)

    return data, data_split


class TqdmStream(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

def setup(args):
    # check some args
    if args.cmd in ['train', 'lr']:
        if args.epochs is None and args.training_steps is None:
            parser.error('Either specify --epochs or --training-steps')
    # configure logging
    if args.debug:
        logging.basicConfig(format='[%(levelname)s  -  %(asctime)s] %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.DEBUG, stream=TqdmStream(sys.stderr))
        logging.debug('Enabled debug mode')
    else:
        logging.basicConfig(format='[%(levelname)s  -  %(asctime)s] %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.INFO, stream=TqdmStream(sys.stderr))
        # logging.basicConfig(filename='log.log',level=logging.DEBUG)

    if args.pre_allocation and not args.nocuda and torch.cuda.is_available():
        # allocate all available memory
        allocate_mem()

    if args.cmd == 'train':
        if not match_datasets(args.data_dir, args.datasets, no_checks=True):
            parser.error('No training dataset(s) have been selected.\nPossible options are: %s' % ', '.join(dataset.name for dataset in DATASETS))
        if args.eval is not None:
            if not match_datasets(args.data_dir, args.eval, no_checks=True):
                parser.error('No evaluation dataset(s) have been selected.\nPossible options are: %s' % ', '.join(dataset.name for dataset in DATASETS))
    elif args.cmd == 'lr':
        if not match_datasets(args.data_dir, args.datasets, no_checks=True):
            parser.error('No training dataset(s) have been selected.\nPossible options are: %s' % ', '.join(dataset.name for dataset in DATASETS))
    elif args.cmd == 'eval':
        if not match_datasets(args.data_dir, args.datasets, no_checks=True):
            parser.error('No evaluation dataset(s) have been selected.\nPossible options are: %s' % ', '.join(dataset.name for dataset in DATASETS))


def _get_latest_tb_run_id(log_dir):
    """
    returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob("{}_[0-9]*".format(log_dir)):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_dir.split('/')[-1] == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def setup_writer(log_dir: str, log_dir_suffix: str = None, purge_step: int = None, debug: bool = False):
    # set up tensorboard logging
    if debug:
        # do not log to tensorboard
        return None

    # tensorboard logger
    data_names = []
    # comment = ''
    # if args.cmd == 'train':
    #     comment += '_train_' + '_'.join(map(str, match_datasets(args.data_dir, args.datasets, no_checks=True)))
    # if args.comment:
    #     comment += '_' + args.comment
    if log_dir_suffix is not None:
        log_dir = log_dir + '/' + str(log_dir_suffix)
    # print(comment)
    # log_id = _get_latest_tb_run_id(log_dir)
    # print(log_id)
    # writer = FileWriter(log_dir=log_dir + "_{}".format(log_id))
    # SummaryWriter(log_dir=log_dir)
    return SummaryWriter(log_dir=log_dir, purge_step=purge_step)


def train(args, seed):
    # set up agent

    agent = MRQAAgent(args.model, args.cache_dir, pretrained_model_dir=args.pretrained_model, disable_cuda=args.nocuda, results=args.results)
    
    # set up tensorboard writer for interactive visualization
    writer = setup_writer(args.logdir, seed, purge_step=agent.training_steps, debug=args.debug)
    agent.add_tb_writer(writer)

    # data
    datasets_train = match_datasets(args.data_dir, args.datasets)
    datasets_eval = match_datasets(args.data_dir, args.eval)
    data_train, data_split = get_datasets(args.data_dir, args.cache_dir, agent.sample_processor, agent.tokenizer, datasets_train, seed=seed, force_preprocess=args.pre_process)
    data_eval, data_split_2 = get_datasets(args.data_dir, args.cache_dir, agent.sample_processor, agent.tokenizer, datasets_eval, seed=seed, force_preprocess=args.pre_process)

    # merge train data
    data_train = reduce(lambda x, y: x + y, data_train)

    # train and evaluate
    metrics = agent.train(data_train, None, data_eval + data_split + data_split_2, args.batch_size, lr=args.learning_rate, num_epochs=args.epochs, num_training_steps=args.training_steps, num_total_training_steps=args.total_training_steps, warmup_ratio=args.warmup_ratio, labels=args.labels, eval_interval=args.eval_interval)
    
    writer.flush()
    return metrics

def eval(args, seed):
    # set up agent
    agent = MRQAAgent(model_dir=args.model, cache_dir=args.cache_dir, disable_cuda=args.nocuda, results=args.results)

    # set up tensorboard writer for interactive visualization
    writer = setup_writer(args.logdir, seed, purge_step=agent.training_steps, debug=args.debug)
    agent.add_tb_writer(writer)

    # data
    datasets = match_datasets(args.data_dir, args.datasets)
    data, data_split = get_datasets(args.data_dir, args.cache_dir, agent.sample_processor, agent.tokenizer, datasets, seed=seed)

    logging.info('Evaluating on %r', ', '.join(map(str, data + data_split)))

    agent.evaluate(data + data_split, args.batch_size)

    writer.flush()


def preprocess(args):
    # preprocess data
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
    sample_processor = SlidingWindowHandler(tokenizer, 512)
    for dataset in match_datasets(args.data_dir, args.datasets):
        get_datasets(args.data_dir, args.cache_dir, sample_processor, tokenizer, {dataset}, force_preprocess=not args.skip_existing)


def low_resource(args, seed):
    # set up agent
    agent = MRQAAgent(args.model, args.cache_dir, pretrained_model_dir=args.pretrained_model, disable_cuda=args.nocuda, results=args.results)
    
    # set up tensorboard writer for interactive visualization
    writer = setup_writer(args.logdir, seed, purge_step=agent.training_steps, debug=args.debug)
    agent.add_tb_writer(writer)

    # data
    datasets = match_datasets(args.data_dir, args.datasets, eval_samples=args.eval_samples)
    data_train, data_eval = get_datasets(args.data_dir, args.cache_dir, agent.sample_processor, agent.tokenizer, datasets, seed=seed)

    # merge train data
    data_train = reduce(lambda x, y: x + y, data_train)

    # train and evaluate
    metrics = agent.train(data_train, None, data_eval, batch_size=args.batch_size, lr=args.learning_rate, num_epochs=args.epochs, num_training_steps=args.training_steps, num_total_training_steps=args.total_training_steps, warmup_ratio=args.warmup_ratio, labels=args.labels, eval_interval=args.eval_interval)
    
    writer.flush()
    return metrics


# from https://stackoverflow.com/a/48774926
def tabulate_events(dirs):
    summary_iterators = [EventAccumulator(dir).Reload() for dir in dirs]

    tags = summary_iterators[0].Tags()['scalars']

    out = defaultdict(list)
    steps = defaultdict(list)
    
    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators if tag in acc.Tags()['scalars']]):
            event_steps = set(e.step for e in events)
            assert len(event_steps) == 1

            out[tag].append([e.value for e in events])
            steps[tag].append(event_steps.pop())

    return out, steps


# from https://stackoverflow.com/a/48774926
def write_events_averaged(writer, d_combined, steps):
    tags, values = zip(*d_combined.items())

    for tag, _values in zip(tags, values):
        for i, mean in zip(steps[tag], numpy.array(_values).mean(axis=-1)):
            writer.add_scalar(tag, mean, global_step=i)

        writer.flush()


if __name__ == "__main__":
    import argparse
    import configparser

    # some parameters might be in ini file
    config = configparser.ConfigParser()
    config.read('config.ini')

    ### parent parsers
    
    # general arguments
    parser_general = argparse.ArgumentParser(description='A parser for the general arguments', add_help=False) # can be used as parent for other parsers
    
    parser_general.add_argument('--data-dir', type=expand_path, required=False if config.get('Paths', 'data_dir', fallback=False) else True, default=config.get('Paths', 'data_dir', fallback='~/.data'), help='the data directory')
    parser_general.add_argument('--cache-dir', type=expand_path, default=config.get('Paths', 'cache_dir', fallback='~/.cache'), help='the cache directory')
    parser_general.add_argument('--nocuda', action='store_true', help='Disables CUDA (otherwise all available GPUs will be used)')
    parser_general.add_argument('-pa', '--pre-allocation', action='store_true', help='Enable pre-allocation of GPU memory (this will allocate 95%% of memory)')
    parser_general.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser_general.add_argument('-s', '--seed', nargs='+', type=int, default=[], help='The seed for the random number generators')
    parser_general.add_argument('--runs', type=check_positive, default=1, help='The number of runs with different seeds')
    parser_general.add_argument('--eval-interval', type=int, default=1000, help='The interval for evaluation (if activated, in training steps)')
    parser_general.add_argument('--pre-process', action='store_true', help='Will force to preprocess any data first')
    
    # logging related arguments
    parser_logging = argparse.ArgumentParser(description='A parser used for logging functionality.', add_help=False) # can be used as parent for other parsers
    
    parser_logging.add_argument('-c', '--comment', type=str, default='', help='Comment for experiment')
    parser_logging.add_argument('-l', '--logdir', type=str, default='runs', help='The log directory for tensorboard (e.g. for structuring experiments)')
    parser_logging.add_argument('-r', '--results', type=str, default='results', help='The file where the results will be stored')

    # training related arguments
    parser_training = argparse.ArgumentParser(description='A parser used for training a model.', add_help=False, parents=[parser_logging]) # can be used as parent for other parsers
    
    parser_training.add_argument('model', type=expand_path, help='the directory where the model and other files are stored (for training & evaluation)')
    parser_training.add_argument('--pretrained-model', type=expand_path, help='a pre-trained model to start off')
    parser_training.add_argument('--epochs', type=check_positive, default=-1, help='number of epochs for training (> 0)')
    parser_training.add_argument('--batch-size', type=check_positive, required=True, help='batch size (> 0)')
    parser_training.add_argument('--training-steps', type=check_positive, default=-1, help='number of total training steps')
    parser_training.add_argument('--total-training-steps', type=check_positive, default=-1, help='number of total training steps')
    parser_training.add_argument('--warmup-ratio', type=float, default=0.1, help='ratio for warmup phase')
    parser_training.add_argument('-lr', '--learning-rate', required=True, type=float, help='optimizer\'s learning rate')
    parser_training.add_argument('--multi-label', action='store_const', default='single', const='multi', dest='labels', help='will use multi label loss function if specified, otherwise single label')
    parser_training.add_argument('--eval-seed', type=int, default=1234, help='the seed for uniformly drawing samples for evaluation')

    # main parser
    parser = argparse.ArgumentParser(description='Run MRQA training and prediction.', parents=[parser_general])
    subparsers = parser.add_subparsers(title='command', dest='cmd', required=True)
    
    ### parser commands

    # train command
    parser_train = subparsers.add_parser('train', description='Runs training on the given datasets.' , parents=[parser_general, parser_training])
    parser_train.set_defaults(func=train)
    
    parser_train.add_argument('datasets', nargs='+', metavar='dataset', help='the dataset(s) used for training the model')
    parser_train.add_argument('--eval', nargs='+', metavar='dataset', help='the dataset(s) used for evaluating the model')
    
    # eval command
    parser_eval = subparsers.add_parser('eval', description='Runs evaluation for the given model on the given datasets.' , parents=[parser_general, parser_logging])
    parser_eval.set_defaults(func=eval)
    
    parser_eval.add_argument('model', type=expand_path, help='the directory where the model and other files are stored (for training & evaluation)')
    parser_eval.add_argument('datasets', nargs='+', metavar='dataset', help='the dataset(s) used for evaluating the model')
    parser_eval.add_argument('--batch-size', type=check_positive, required=True, help='batch size (> 0)')
    parser_eval.add_argument('--eval-seed', type=int, default=1234, help='the seed for uniformly drawing samples for evaluation')
    
    # lr command
    parser_lowresource = subparsers.add_parser('lr', description='Runs low-resource experiments for the given domain where some samples are used for training and some for evaluation.' , parents=[parser_general, parser_training])
    parser_lowresource.set_defaults(func=low_resource)

    parser_lowresource.add_argument('datasets', nargs='+', metavar='dataset', help='the dataset(s) used for training and evaluating the model')
    parser_lowresource.add_argument('--eval-samples', type=int, default=500, help='the number of samples used for evaluation')
    
    # preprocess command
    parser_preprocess = subparsers.add_parser('preprocess', description='Runs preprocessing for the given datasets.' , parents=[parser_general])
    parser_preprocess.set_defaults(func=preprocess)

    parser_preprocess.add_argument('datasets', nargs='+', metavar='dataset', help='the dataset(s) to preprocess')
    parser_preprocess.add_argument('--skip-existing', action='store_true', help='preprocesses the given data')

    args = parser.parse_args()
    # print(args)

    setup(args)

    if args.cmd in ['train', 'eval', 'lr']:
        metrics_all = {}
        seeds = []
        for run in range(args.runs):
            logging.info("===== Run %d/%d =====", run+1, args.runs)
            # set seed
            seed = init_random(args.seed[run] if len(args.seed) > run else None)
            seeds.append(seed)

            # call command-specific function
            update_dict(metrics_all, {seed: args.func(args, seed)})

        if args.runs > 1:
            # plot average over seeds to tensorboard

            d, steps = tabulate_events([os.path.join(args.logdir, str(seed)) for seed in seeds])
            writer = setup_writer(args.logdir, 'avg', purge_step=-1)
            write_events_averaged(writer, d, steps)

            
    elif args.cmd in ['preprocess']:
        # call command-specific function
        args.func(args)

# general imports 
import torch
from torch.utils.data import DataLoader
import numpy 
from transformers import PreTrainedTokenizer
from typing import Set, Union
from functools import reduce
import re 
from dataclasses import replace
import os
import urllib


from data_preprocessing.data import (BertQASampler, MRQADataset, pad_batch, Dataset, SharedTaskDatasetReader)
from data_preprocessing.agent import MRQAAgent


reader = SharedTaskDatasetReader(answer_first_occurence_only = True)

DATASETS = [
Dataset('SQuAD-train', 'train/SQuAD.jsonl.gz', reader),
Dataset('SQuAD-dev', 'dev/SQuAD.jsonl.gz', reader)
]


def get_datasets(data_dir: str, cache_dir: str, sample_processor: callable, tokenizer: PreTrainedTokenizer, datasets: Set[Dataset], seed: int = None, force_preprocess: bool = False):
    data, data_split = [], []

    for dataset in datasets:
        _data = MRQADataset.load(data_dir, cache_dir, dataset, sample_processor, tokenizer, force_preprocess=force_preprocess)
        num_samples_eval = dataset.num_samples_eval
        if num_samples_eval is not None:
            rng = numpy.random.RandomState(eval_seed)
            data_split.append(_data.sample_split(num_samples_eval, shuffle_fn=rng.shuffle, remove_samples=True, seed=eval_seed))
        if dataset.num_samples > 0:
            data.append(_data.sample_split(dataset.num_samples, seed=seed))
        elif dataset.num_samples == -1:
            data.append(_data)

    return data, data_split

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
                        continue

                datasets.add(replace(dataset, num_samples=num_samples, num_samples_eval=num_samples_eval))
    return datasets


eval_seed = 1234
seed = 3957113738
model="models"
cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'cache')
pretrained_model = None
nocuda = False
results = "results"
datasets = ['SQuAD-train']
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'datasets')
pre_process = False

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloader(batch_size):

    urllib.request.urlretrieve("https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz", filename=os.path.join(data_dir, 'train', 'SQuAD.jsonl.gz'))

    agent = MRQAAgent(model, cache_dir, pretrained_model_dir=pretrained_model, disable_cuda=nocuda, results=results)
    datasets_train = match_datasets(data_dir, datasets) # returns just a set where the Dataset is inside!
    data_train, data_split = get_datasets(data_dir, cache_dir, agent.sample_processor, agent.tokenizer, datasets_train, seed=seed, force_preprocess=pre_process)
    # merge train data (When more training datasets are used --> creates a single MEQADataset class)
    data_train = reduce(lambda x, y: x + y, data_train)


    batch_sampler = BertQASampler(data_source=data_train, batch_size=batch_size, training=True, shuffle=True, drop_last=False, fill_last=True, repeat=True)
    batch_sampler_iterator = iter(batch_sampler)
    dataloader = DataLoader(data_train, batch_sampler=batch_sampler_iterator, collate_fn=pad_batch)


    return dataloader




# general imports 
import torch
from torch.utils.data import DataLoader
import numpy 
from transformers import BertTokenizer, PreTrainedTokenizer
from typing import Set, Union
from functools import reduce
import re 
from dataclasses import replace
from collections import Counter, defaultdict
from operator import attrgetter
import sys 
import os

# change this to the location where your files from mrqa-basline are placed
#sys.path.append("/Users/maxkeller/Documents/Uni/Softwaretechnik/Projektarbeiten/mrqa-baseline/modAL_prototype_test/get_bert_data_test")
# special imports from maximilians code 
from data import (BertQASampler, MRQADataset, SlidingWindowHandler,
                  normalize_answer, pad_batch, Dataset, SharedTaskDatasetReader)
from agent import MRQAAgent


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
    dataset_names_duplicates = [name for name, count in Counter(map(attrgetter('name'), datasets)).items() if count > 1]
    return datasets


eval_seed = 1234
seed = 3957113738
model="models"
cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../cache')
pretrained_model = None
nocuda = False
results = "results"

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../datasets')
pre_process = False

"""
training_steps = 10
train_steps = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

#maybe later
# set up tensorboard writer for interactive visualization
writer = setup_writer(args.logdir, seed, purge_step=agent.training_steps, debug=args.debug)
agent.add_tb_writer(writer)
"""


# loads the dataset into data_train, data_split is not further needed 
"""
In the metadata there is the full SQAD sample
additional we will hava a: 
mask (1, were there is real data, 0 when it is just a padding),
segments, 
label_multi, 
segments, 
wordpiece_to_token_idx, 
token_to_context_idx, 
input, 
label (Maybe token-ids but I do not know this in detail ...), 
"""

def get_dataloader(datasets, batch_size):
    agent = MRQAAgent(model, cache_dir, pretrained_model_dir=pretrained_model, disable_cuda=nocuda, results=results)
    datasets_train = match_datasets(data_dir, datasets) # returns just a set where the Dataset is inside!
    data_train, data_split = get_datasets(data_dir, cache_dir, agent.sample_processor, agent.tokenizer, datasets_train, seed=seed, force_preprocess=pre_process)
    # merge train data (When more training datasets are used --> creates a single MEQADataset class)
    data_train = reduce(lambda x, y: x + y, data_train)


    #data_train: MRQADataset # needs still to be added

    batch_sampler = BertQASampler(data_source=data_train, batch_size=batch_size, training=True, shuffle=True, drop_last=False, fill_last=True, repeat=True)
    batch_sampler_iterator = iter(batch_sampler)
    dataloader = DataLoader(data_train, batch_sampler=batch_sampler_iterator, collate_fn=pad_batch)

    #data_iter = iter(dataloader) # create iterator so that the same can be used in all function calls (also working with zip)

    return dataloader




import errno
import gzip
import json
import logging
import math
import os
import pickle
import random
import re
import string
from collections import Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from operator import attrgetter, itemgetter
from typing import Any, Callable, Dict, Generator, Iterable, List, Union

import jsonlines
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizer

from utils import print_mem_usage


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    reader: Callable
    num_samples: int = -1
    num_samples_eval: Union[None, int] = None

    def __str__(self):
        return self.name

    def get_total_samples(self, data_dir):
        with gzip.GzipFile(os.path.join(data_dir, self.path), 'r') as fin:
            reader = jsonlines.Reader(fin)
            
            num_context = 0
            num_samples = 0

            for obj in reader:
                if 'header' in obj:
                    # skip the header
                    continue

                num_context += 1
                num_samples += len(obj['qas'])
        return num_samples


RE_REMOVE_PUNCTUATION = re.compile(r'[%s]' % string.punctuation)
RE_REMOVE_ARTICLES = re.compile(r'\b(a|an|the)\b')
RE_REMOVE_WHITESPACE = re.compile(r'[\s]+')

def normalize_answer(answer: str):
    def lower(answer: str):
        return answer.lower()

    def remove_punctuation(answer: str):
        return re.sub(RE_REMOVE_PUNCTUATION, '', answer)

    def remove_articles(answer: str):
        return re.sub(RE_REMOVE_ARTICLES, '', answer)

    def remove_whitespace(answer: str):
        return re.sub(RE_REMOVE_WHITESPACE, ' ', answer.strip())

    return remove_whitespace(remove_articles(remove_punctuation(lower(answer))))

def normalize_answer_allen(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def pad_batch(batch):
    # TODO add CUDA support (i.e. consider device)
    # pad input before collating the batch
    max_len = max([sample['input'].size(0) for sample in batch])
    for sample in batch:
        sample_len = sample['input'].size(0)
        sample['input'] = torch.cat((sample['input'], torch.zeros(max_len - sample_len, dtype=torch.long)), dim=0)
        sample['segments'] = torch.cat((sample['segments'], torch.ones(max_len - sample_len, dtype=torch.long)), dim=0) # pad with segment B emnedding
        sample['mask'] = torch.cat((sample['mask'], torch.zeros(max_len - sample_len, dtype=torch.long)), dim=0)
        # pad multi label vector
        sample['label_multi'] = torch.cat((sample['label_multi'], torch.zeros(2, max_len - sample_len, dtype=torch.long)), dim=1)

    # collate only some keys using default collate
    collated_batch = default_collate([{k: v for k,v in sample.items() if k in ['input', 'segments', 'mask', 'label', 'label_multi']} for sample in batch])

    for sample in batch:
        for key in sample:
            if key not in ['input', 'segments', 'mask', 'label', 'label_multi']:
                if key == 'metadata':
                    if key not in collated_batch:
                        collated_batch[key] = {}
                    for _key in sample[key]:
                        if _key not in collated_batch[key]:
                            collated_batch[key][_key] = []
                        collated_batch[key][_key].append(sample[key][_key])
                else:
                    if key not in collated_batch:
                        collated_batch[key] = []
                    collated_batch[key].append(sample[key])

    return collated_batch


class BertQASamplerIterator(Iterator):
    def __init__(self, data_source, batch_size, training, shuffle, drop_last: bool = False, fill_last: bool = False, repeat: bool = False, return_epoch: bool = False):
        self.data = data_source
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle

        if drop_last and fill_last:
            raise ValueError('drop_last is mutually exclusive with fill_last')

        self.fill_last = fill_last
        self.drop_last = drop_last
        self.repeat = repeat

        self.epoch = 0
        self.return_epoch = return_epoch
        self.iterator = self._iterator()
        self.saved_items = []

        self.stop = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration

        if not self.training:
            # for evaluation we do not apply special handling because items can exceed the batch_size anyway
            if self.return_epoch:
                return next(self.iterator), self.epoch
            else:
                items = next(self.iterator)
                return items
        else:
            items = self.saved_items[:self.batch_size]
            self.saved_items = self.saved_items[self.batch_size:]
            epoch = 0
            while len(items) < self.batch_size:
                try:
                    self.saved_items += next(self.iterator)
                except StopIteration as e:
                    if self.repeat:
                        if self.drop_last:
                            items = []
                        # create new iterator
                        self.epoch += 1
                        self.iterator = self._iterator()
                    else:
                        if self.drop_last:
                            raise StopIteration
                        if self.fill_last:
                            # set flag to yield only one more batch
                            self.stop = True
                            # create new iterator
                            self.epoch += 1
                            self.iterator = self._iterator()
                        else:
                            if len(items) == 0:
                                raise StopIteration
                            break # end while loop here

                # add elements (until batch size is reached or all items from saved_items have been consumed)
                items.extend(self.saved_items.pop(0) for _ in range(min(self.batch_size - len(items), len(self.saved_items))))

            # print(self.epoch + epoch, items)
            if self.return_epoch:
                return items, self.epoch + epoch
            else:
                return items
                
    def _iterator(self) -> Generator[List, None, None]:
        if self.shuffle:
            data_indices = random.sample(range(len(self.data)), k=len(self.data))
        else:
            data_indices = range(len(self.data))


        if self.training:
            # for training we use a random chunk for each instance which actually contains the answer
            cur_batch = []
            for idx in data_indices:
                samples_with_answer = [sample_idx for sample_idx, sample in enumerate(self.data[idx]) if sample['metadata']['has_answer']]
                cur_batch.append((idx, random.choice(samples_with_answer)))
                # NOTE: answer always occurs in context in MRQA datasets
                # TODO extend to also include samples without answer -> change model output
                if len(cur_batch) == self.batch_size:
                    yield cur_batch
                    cur_batch = []
            
            if cur_batch:
                # yield remainder
                yield cur_batch
        else:
            # make sure to have all instances of one context in same batch (context has to be split sometimes since Bert input has max length of 512 but postprocessing will collapse predictions (while not training) into single answer)
            cur_batch = []
            for idx in data_indices:
                num_instance_chunks = len(self.data[idx])
                instance_indices = list(zip(num_instance_chunks * [idx], range(num_instance_chunks)))

                # puts chunks of same instance into same batch but batches might exceed batch_size
                if cur_batch and len(cur_batch) + len(instance_indices) > self.batch_size:
                    # print("Batch indices:", cur_batch)
                    yield cur_batch
                    cur_batch = instance_indices
                else:
                    cur_batch += instance_indices
            
                # yields batches of exactly batch_size (but last batch)
                # offset = 0
                # for _ in range(math.ceil((num_instance_chunks + len(cur_batch))/self.batch_size)):
                #     num_add_instance_chunks = self.batch_size-len(cur_batch)
                #     cur_batch += instance_indices[offset:offset+num_add_instance_chunks]
                #     offset += num_add_instance_chunks
                #     if len(cur_batch) == self.batch_size:
                #         # print("Batch indices:", cur_batch)
                #         yield cur_batch
                #         cur_batch = []
            if cur_batch:
                # yield remainder
                # print("Batch indices:", cur_batch)
                yield cur_batch


class BertQASampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, training, shuffle, drop_last: bool = False, fill_last: bool = False, repeat: bool = False):
        self.data = data_source
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle

        if drop_last and fill_last:
            raise ValueError('drop_last is mutually exclusive with fill_last')

        self.fill_last = fill_last
        self.drop_last = drop_last
        self.repeat = repeat

    def _iter(self, return_epoch: bool = False):
        return BertQASamplerIterator(self.data, self.batch_size, self.training, self.shuffle, self.drop_last, self.fill_last, self.repeat, return_epoch=return_epoch)

    def __iter__(self):
        return self._iter()

    def __len__(self):
        if self.training:
            # return amount of batches: only one instance per context is chosen for training
            if self.repeat:
                # runs infinitely
                return -1
            if self.drop_last:
                # cut off last batch if it's not full
                return len(self.data)//self.batch_size
            # round up to incude last batch
            return math.ceil(len(self.data)/self.batch_size)
        else:
            # return amount of samples: all instances are used -> length is impossible to determine since it depends on the order of the samples
            # return math.ceil(sum([len(sample) for sample in self.data])/self.batch_size)
            return -1

    def get_epochs(self):
        # return a generator yielding the epoch for increasing batches
        # here we make use of the real generator but return the epoch instead of the items
        iterator = self._iter(return_epoch=True)
        yield from map(itemgetter(1), iterator)


@dataclass
class MRQADataset(torch.utils.data.Dataset):
    def __init__(self, data, *datasets):
        self.data = data
        self.datasets = datasets

    def __add__(self, other: 'MRQADataset'):
        return self.__class__(self.data + other.data, *self.datasets, *other.datasets)

    def extend(self, other: 'MRQADataset'):
        return self + other

    def __str__(self):
        return '%s (%d sample(s))' % (self.identifier, len(self))

    @property
    def identifier(self):
        return '_'.join(map(attrgetter('name'), self.datasets))

    @classmethod
    def load(cls, root_dir: str, cache_dir: str, dataset: Dataset, handler, tokenizer: BertTokenizer, stride: int = 128, force_preprocess: bool = False, force_post_check: bool = False):
        """
        Args:
            root_dir (string): Directory with all the datasets.
            cache_dir (string): Directory where files will be cached (for example preprocessed data).
            dataset (Dataset): The dataset to process.
            tokenizer: The tokenizer used for preprocessing the data.
            train (boolean, optional): Whether to load train or dev dataset, default is True.
            force_preprocess (boolean, optional): Force preprocessing of data instead of loading.
        """
        
        preprocess = False
        cache_filepath = os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), dataset.name + '.bin')
        
        if not force_preprocess:
            # try loading preprocessed data
            if os.path.exists(cache_filepath):
                # file exists -> check if we should continue loading it
                if datetime.strptime(json.load(open('version', 'r'))['preprocessed_data'], '%d/%m/%y %H:%M:%S') > datetime.fromtimestamp(os.path.getmtime(cache_filepath)):
                    # preprocessed data is outdated
                    logging.info("Did not load preprocessed data for dataset %s: outdated", dataset.name)
                    preprocess = True
                else:
                    # load preprocessed data
                    try:
                        with open(cache_filepath, 'rb') as cache_file:
                            data = pickle.load(cache_file)
                            if force_post_check:
                                cls.post_check(data, tokenizer)
                        logging.info("Loaded preprocessed data for dataset %s", dataset.name)
                    except FileNotFoundError:
                        logging.info("Did not load preprocessed data for dataset %s: not found", dataset.name)
                        preprocess = True
            else:
                # file does not exist
                preprocess = True

        if force_preprocess or preprocess:
            logging.info("Preprocessing data for dataset %s", dataset.name)
            # load data, preprocess and save
            filepath = os.path.join(os.path.abspath(os.path.expanduser(root_dir)), dataset.path)
            data = []
            for obj in dataset.reader(filepath, tokenizer):
                instances = []
                # process each sample
                for instance in handler(**obj):
                    instance['metadata']['sample_id'] = len(data)
                    instance['metadata']['dataset'] = dataset.name
                    instances.append(instance)
                if instances:
                    # add amount of instances for the sample
                    num_instances = len(instances)
                    for instance in instances:
                        instance['metadata']['num_instances'] = num_instances
                    data.append(instances)
            
            cls.post_check(data, tokenizer)

            # create directories if necessary
            if not os.path.exists(os.path.dirname(cache_filepath)):
                try:
                    os.makedirs(os.path.dirname(cache_filepath))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(cache_filepath, 'wb') as cache_file:
                pickle.dump(data, cache_file)

        return cls(data, dataset)

    def sample_split(self, n: int, shuffle_fn: Callable = random.shuffle, remove_samples: bool = False, seed: int = None):
        # copy list
        _data = self.data[:]
        # shuffle data in place
        shuffle_fn(_data)
        # create new dataset with n samples
        datasets = self.datasets
        if seed is not None:
            datasets = {replace(dataset, name=dataset.name + ' (split: %d - seed %d)' % (n, seed)) for dataset in datasets}
        data_split = self.__class__(_data[:n], *datasets)
        
        if remove_samples:
            self.data = _data[n:]

        return data_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # expect index to be an int or a tuple of context, instance
        if isinstance(idx, int):
            return self.data[idx]
        else:
            # here idx should be a tuple of size 2
            return self.data[idx[0]][idx[1]]

    @staticmethod
    def post_check(_data, tokenizer):
        for instances in _data:
            for instance in instances:
                id = f"{instance['metadata']['dataset']} - id: {instance['metadata']['id']} - qid: {instance['metadata']['qid']}"
                input_token_ids: torch.FloatTensor = instance['input']
                # if input_token_ids[0] != self.tokenizer.cls_token_id:
                #     logging.error("Input does not start with CLS token (%s)" % id)
                #     logging.error(f"Tokens: {self.tokenizer.convert_ids_to_tokens(instance['input'])}")
                # if input_token_ids[-1] != self.tokenizer.sep_token_id:
                #     logging.error("Input does not end with SEP token (%s)" % id)
                #     logging.error(f"Tokens: {self.tokenizer.convert_ids_to_tokens(instance['input'])}")
                # if input_token_ids[len(instance['metadata']['question_tokens']) + 1] != self.tokenizer.sep_token_id:
                #     logging.error("Input does not contain SEP token after question (%s)" % id)
                #     logging.error(f"Tokens: {self.tokenizer.convert_ids_to_tokens(instance['input'])}")
                # if (input_token_ids == self.tokenizer.cls_token_id).sum() != 1:
                #     logging.error("Input does not contain exactly one CLS token (%s)" % id)
                #     logging.error(f"Tokens: {self.tokenizer.convert_ids_to_tokens(instance['input'])}")
                # if (input_token_ids == self.tokenizer.sep_token_id).sum() != 2:
                #     logging.error("Input does not contain exactly two SEP tokens (%s)" % id)
                #     logging.error(f"Tokens: {self.tokenizer.convert_ids_to_tokens(instance['input'])}")
                
                # check for UNK token in input
                # if (input_token_ids == self.tokenizer.unk_token_id).sum() != 0:
                #     logging.error("Input contains UNK token (%s)" % id)
                #     logging.error(f"Tokens: {self.tokenizer.convert_ids_to_tokens(instance['input'])}")

                # if instance['metadata']['has_answer']:
                #     answer_span = instance['label'] + instance['metadata']['context_instance_start_end'][0]
                #     extracted_answer = normalize_answer(instance['metadata']['context'][instance['metadata']['token_to_context_idx'][instance['metadata']['wordpiece_to_token_idx'][answer_span[0].item()]][0]:instance['metadata']['token_to_context_idx'][instance['metadata']['wordpiece_to_token_idx'][answer_span[1].item()]][1]])
                #     extracted_answer2 = normalize_answer_allen(instance['metadata']['context'][instance['metadata']['token_to_context_idx'][instance['metadata']['wordpiece_to_token_idx'][answer_span[0].item()]][0]:instance['metadata']['token_to_context_idx'][instance['metadata']['wordpiece_to_token_idx'][answer_span[1].item()]][1]])
                #     answers = [normalize_answer(answer) for answer in instance['metadata']['answers_per_instance']]
                #     answers2 = [normalize_answer_allen(answer) for answer in instance['metadata']['answers_per_instance']]
                #     if extracted_answer not in answers:
                #         logging.error("Answer span does not match the answer (%s)" % id)
                #         logging.error("Extracted span is '%s' and correct answers are '%s'" % (extracted_answer, answers))
                #         print(list(enumerate(instance['metadata']['cur_instance_context_tokens'])))
                #         print(answer_span)
                #         print(extracted_answer)
                #         print(extracted_answer2)
                #         print(answers)
                #         print(answers2)

                #         exit()


class SlidingWindowHandler():
    def __init__(self, tokenizer, max_size: int, stride: int = 128):
        self.max_size = max_size
        self.stride = stride # stride for sliding window for splitting context into multiple chunks
        self.tokenizer = tokenizer

    def __call__(self, context_tokens, question_tokens, answers, metadata, **kwargs: dict):
        # split context using sliding window
        # this has to be done for each question individually since lengts of questions may differ
        len_context = len(context_tokens)
        len_question = len(question_tokens)

        qid = metadata.get('qid')

        if len_question > self.max_size - 3:
            logging.warning(f"Skipping question{f' (id {qid})' if qid is not None else ''} since its length is already bigger than the allowed Bert input size!")
            raise StopIteration
        elif len_question > 450: # subtracting [CLS] toking here
            logging.warning(f"Question{f' (id {qid})' if qid is not None else ''} has already {len_question} tokens and it is unlikely to give correct answers since the context has to be cut short.")
            
        cur_instance_context_offset = 0
        context_tokens_max_len = self.max_size - len_question - 3 # -1 for CLS token and -2 for SEP tokens
        while (cur_instance_context_offset == 0) or (cur_instance_context_offset + context_tokens_max_len < len_context + self.stride):
            # process all context tokens

            # correct context tokens (skip SEP tokens at the beginning)
            cur_instance_context_tokens_len = min(context_tokens_max_len, len_context - cur_instance_context_offset)
            cur_instance_context_end = cur_instance_context_offset + cur_instance_context_tokens_len
            skip_tokens = 0
            # for token in context_tokens[cur_instance_context_offset:cur_instance_context_end + 1]:
            #     if token == '[SEP]':
            #         skip_tokens += 1
            #     else:
            #         break
            cur_instance_context_offset += skip_tokens
            
            # check for subwords at the end of the context
            cur_instance_context_tokens_len = min(context_tokens_max_len, len_context - cur_instance_context_offset)
            num_omit_tokens = 0
            if len_context > cur_instance_context_offset + cur_instance_context_tokens_len:
                # last token of current instance is not last token of context -> check if we have to omit tokens at the end due to WordPiece tokenization
                if context_tokens[cur_instance_context_offset + cur_instance_context_tokens_len].startswith('##'):
                    # next token after last current token is within token -> remove all WordPiece tokens belonging to this token
                    num_omit_tokens = 1
                    while num_omit_tokens < cur_instance_context_tokens_len:
                        if not context_tokens[cur_instance_context_offset + cur_instance_context_tokens_len - num_omit_tokens].startswith('##'):
                            break
                        num_omit_tokens += 1

            cur_instance_context_end = cur_instance_context_offset + cur_instance_context_tokens_len - num_omit_tokens - 1
            cur_instance_context_tokens = context_tokens[cur_instance_context_offset:cur_instance_context_end + 1]
            # print(q['question'])
            # print(context_wordpiece_tokens)
            # print(cur_instance_context_tokens)
            # exit()
            # TODO use tokenizer.build_inputs_with_special_tokens method
            instance_input = torch.tensor(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token] + cur_instance_context_tokens + [self.tokenizer.sep_token]))
            if instance_input.size(0) > self.max_size:
                logging.error(f"Something went wrong: instance has more than {self.max_size} tokens after splitting into chunks")
                
            # add metadata dict and update it with some values
            instance = {
                'metadata': metadata.copy(),
                'wordpiece_to_token_idx': kwargs.get('wordpiece_to_token_idx'),
                'token_to_context_idx': kwargs.get('token_to_context_idx'),
                'input': instance_input,
                'segments': torch.LongTensor([0] * (len_question + 2) + [1] * (len(cur_instance_context_tokens) + 1)),
                'mask': torch.LongTensor([1] * instance_input.size(0))
                }
            # add kwargs to metadata
            instance['metadata'].update(kwargs) # TODO at this point 'wordpiece_to_token_idx' and 'token_to_context_idx' are added again (now to the 'metadata' dict) since they are still in the kwargs
            instance['metadata'].update({
                'context_tokens': context_tokens,
                'question_tokens': question_tokens,
                'original_answers': answers, # list of tuple of (answer, span)
                'cur_instance_context_tokens': cur_instance_context_tokens,
                'context_instance_start_end': torch.LongTensor((cur_instance_context_offset, cur_instance_context_end)),
                'length': instance_input.size(0)
            })

            span_offset = len_question + 2 - cur_instance_context_offset
            # label_start_idx = answer_span[0] + span_offset
            # label_end_idx = answer_span[1] + span_offset
            # print(answer_span)
            # print(span_offset)
            # exit()

            # if instance['metadata']['qid'] == '94ca9bf4ee72459085a2b6464d531562':
            #     print("===========")
            #     print(instance_input.size(0))
            #     print(*answer_span)
            #     print(cur_instance_context_offset)
            #     print(skip_tokens)
            #     print(cur_instance_context_tokens_len)
            #     print(context_tokens_max_len)
            #     print(len_context - cur_instance_context_offset)
            #     print(cur_instance_context_tokens_len + len(question_tokens) + 3)
            #     print(len(cur_instance_context_tokens))
            #     print(label_start_idx, label_end_idx)
            #     print((['[CLS]'] + question_tokens + ['[SEP]'] + cur_instance_context_tokens)[label_start_idx:label_end_idx+1])
            #     print(kwargs['answer_from_span'])
            #     print(metadata['answers_per_instance'])
            #     print(0 <= label_start_idx < len(cur_instance_context_tokens) + len_question + 2, 0 <= label_end_idx < len(cur_instance_context_tokens) + len_question + 2)
            label_multi = torch.zeros(2, instance_input.size(0), dtype=torch.long)
            for answer, span in answers:
                label_start_idx = span[0] + span_offset
                label_end_idx = span[1] + span_offset
                assert label_start_idx <= label_end_idx, f"Start label index occurs after end label index:\n{instance}\n{list(enumerate(context_tokens))}"
                
                if 0 <= label_start_idx < len(cur_instance_context_tokens) + len_question + 2 and 0 <= label_end_idx < len(cur_instance_context_tokens) + len_question + 2:
                    # only consider span if start and end are within context
                    label_multi[0][label_start_idx] = 1
                    label_multi[1][label_end_idx] = 1

            instance['label_multi'] = label_multi
            # TODO do we need an indicator for current context having label_multi set or is `has_answer` sufficient?

            span = answers[0][1]
            label_start_idx = span[0] + span_offset
            label_end_idx = span[1] + span_offset

            if 0 <= label_start_idx < len(cur_instance_context_tokens) + len_question + 2 and 0 <= label_end_idx < len(cur_instance_context_tokens) + len_question + 2:
                # answer is in current context chunk
                # print(*answer_span)
                # print(label_start_idx, label_end_idx)
                # print(cur_instance_context_tokens[label_start_idx:label_end_idx+1])
                # exit()
                instance['label'] = torch.LongTensor([label_start_idx, label_end_idx])
                instance['metadata']['has_answer'] = True
            else:
                # answer is not within current context chunk
                instance['label'] = torch.LongTensor([-1, -1])
                instance['metadata']['has_answer'] = False

            cur_instance_context_offset += self.stride
            yield instance
        # if instance['metadata']['qid'] == '94ca9bf4ee72459085a2b6464d531562':
        #     exit()


class SharedTaskDatasetReader():
    def __init__(self, answer_first_occurence_only: bool = True):
        self.answer_first_occurence_only = answer_first_occurence_only

    def __call__(self, filepath: str, tokenizer):
        # directly read from gzip file
        with gzip.GzipFile(filepath, 'r') as fin:
            reader = jsonlines.Reader(fin)

            for obj in reader:
                if 'header' in obj:
                    # skip the header
                    continue

                i = 0
                for instance in self.process_sample(obj, tokenizer):
                    i += 1
                    yield instance
                # assert i == sum(len(qa['detected_answers']) for qa in obj['qas']), "Not all instances have been returned!"
        

    def process_sample(self, sample: Dict[str, Any], tokenizer):
        # applying WordPiece tokenizer manually in order to be able to map the label from tokens (after basic splitting; already done in MRQA data) to WordPiece tokens
        context_wordpiece_tokens = []
        token_to_wordpiece_idx = []
        wordpiece_to_token_idx = []
        token_to_context_idx = []
        num_wordpiece_tokens = 0
        last_token = None
        context_token_offset = 0

        for token_idx, token in enumerate(sample['context_tokens']):
            # store mapping from token idx to context idx
            token_to_context_idx.append((token[1], token[1] + len(token[0])))
            token = token[0]
            # TODO handle splitted tokens in context

            # replace PAR, TLE & DOC token with SEP token
            if token in ['[DOC]', '[TLE]', '[PAR]']:
                token = '[SEP]'

            # TODO remove special tokens
            # TODO make sure to handle offset of tokens to original tokens for answer span correction
            # if token == last_token and token in ['[SEP]', '[CLS]']:
            #     context_token_offset += 1
            #     continue
            # last_token = token

            wordpiece_tokens = tokenizer.tokenize(token)
            context_wordpiece_tokens += wordpiece_tokens
            num_new_wordpiece_tokens = len(wordpiece_tokens)
            # store mapping from token idx to wordpiece idx
            token_to_wordpiece_idx.append((num_wordpiece_tokens, num_wordpiece_tokens + num_new_wordpiece_tokens - 1)) # TODO make end exclusive (remove -1)
            # store mapping from wordpiece idx to token idx
            for idx in range(num_wordpiece_tokens, num_wordpiece_tokens + num_new_wordpiece_tokens):
                wordpiece_to_token_idx.append(token_idx)
            num_wordpiece_tokens += num_new_wordpiece_tokens
            
        # print(len(sample['context_tokens']))
        for q in sample['qas']:
            question_tokens = tokenizer.tokenize(q['question'])

            # store all answers per question
            metadata = {
                'context': sample['context'],
                'paragraph': ' [CLS] ' + q['question'] + ' [SEP] ' + sample['context'] + ' [SEP] ',
                'question': q['question'],
                'context_char_offset': len(q['question']) + 14,
                'answers_per_instance': list(set(q['answers'])), # remove duplicates
                'id': sample.get('id'),
                'qid': q['qid'],
            }

            answers = []
            for qa in q['detected_answers']:
                answers.extend((qa['text'], (token_to_wordpiece_idx[span[0]][0], token_to_wordpiece_idx[span[1]][1])) for span in qa['token_spans'])
                
            # always add answer no matter whether it is part of the current chunk
            instance = {
                'context_tokens': context_wordpiece_tokens,
                'context_tokens_original': sample['context_tokens'],
                'question_tokens': question_tokens,
                'question_tokens_original': q['question_tokens'],
                'answers': answers,
                'wordpiece_to_token_idx': wordpiece_to_token_idx,
                'token_to_context_idx': token_to_context_idx,
                'metadata': metadata,
                'answers_from_span': {tokenizer.convert_tokens_to_string(context_wordpiece_tokens[span[0]:span[1] + 1]) for _, span in answers},
            }

            yield instance


if __name__ == "__main__":
    ROOT_DIR = '~/arbeitsdaten/data/qa'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # print memory usage
    print_mem_usage()

    data = MRQADataset(Dataset.SQuAD, ROOT_DIR, tokenizer)
    print("Num samples:", len(data))

    # print memory usage
    print_mem_usage()

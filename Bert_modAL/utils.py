import errno
import logging
import os
import random

import numpy
import psutil
import torch
import yaml

GLOBAL_SEED = None


def init_random(seed: int = None):
    """
    Initializes the random generators to allow seeding.
    Args:
        seed (int): The seed used for all random generators.
    """
    global GLOBAL_SEED  # pylint: disable=global-statement

    if seed is None:
        tmp_random = numpy.random.RandomState(None)
        GLOBAL_SEED = tmp_random.randint(2**32-1, dtype='uint32')
    else:
        GLOBAL_SEED = seed

    # initialize random generators
    numpy.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    try:
        # try to load torch and initialize random generator if available
        import torch
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # gpu
        torch.manual_seed(GLOBAL_SEED)  # cpu
    except ImportError:
        pass

    try:
        # try to load tensorflow and initialize random generator if available
        import tensorflow
        tensorflow.random.set_seed(GLOBAL_SEED)
    except ImportError:
        pass

    logging.info("Seed is %d", GLOBAL_SEED)
    return GLOBAL_SEED


def print_mem_usage():
    process = psutil.Process(os.getpid())
    bytes_in_memory = process.memory_info().rss
    print(f"Current memory consumption: {bytes_in_memory/1024/1024:.0f} MiB")
    # print(f"Current memory consumption:\nBytes: {bytes_in_memory} - Kibibytes: {bytes_in_memory/1024} - Mebibytes: {bytes_in_memory/1024/1024} - Gibibytes: {bytes_in_memory/1024/1024/1024}")


def check_mem():
    mem = [gpu.split(",") for gpu in os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split('\n')]
    return mem


def allocate_mem():
    # this will allocate GPU memory
    # the allocator is cached until torch.cuda.empty_cache() is called (or the program ends)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        try:
            gpus = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].strip('[]').split(',')))
        except ValueError:
            # can only occur if it hasn't been checked for torch.cuda.is_available()
            gpus = range(torch.cuda.device_count())
    else:
        gpus = range(torch.cuda.device_count())

    gpu_mem = check_mem()
    for gpu_id, gpu in enumerate(gpus):
        total, used = map(int, gpu_mem[gpu])
        
        max_mem = total * 0.9 # somehow allocatable memory is always lower than total memory
        block_mem = int((max_mem - used) * 1024 * 1024 / 4) # from float to MiB // one float is 4 byte
        if block_mem >= 0:
            x = torch.empty(block_mem, device=torch.device('cuda:%d' % gpu_id))
            del x # actually not necessary as pointer is removed once function returns (and gc runs)
        else:
            logging.warning('Cannot allocate memory on gpu %d: maximum exceeded' % gpu)


def extract_span(start_logits, end_logits, batch):
    context_instance_start_end = batch['metadata']['context_instance_start_end']
    context_wordpiece_tokens = batch['metadata']['cur_instance_context_tokens']
    # TODO use given lengths
    # lengths = torch.tensor([len(context) for context in context_wordpiece_tokens])
    # inputs are batch x sequence
    # TODO maybe vectorize computation of best span?
    num_samples = start_logits.size(0)
    scores = [None] * num_samples
    spans = [None] * num_samples
    answers = [None] * num_samples
    max_score_start, max_spans_start = start_logits.max(dim=1)
    for sample_id in range(num_samples):
        # make sure that span start is within context
        score_start, span_start = start_logits[sample_id][len(batch['metadata']['question_tokens'][sample_id]) + 2:batch['metadata']['length'][sample_id] - 1].max(dim=0)
        span_start += len(batch['metadata']['question_tokens'][sample_id]) + 2
        # context_token_offset = context_instance_start_end[sample_id][0]
        # make sure that end is after start and within context
        # print(end_logits[sample_id])
        # print(end_logits[sample_id][span_start:batch['metadata']['length'][sample_id] - 1])
        if span_start == batch['metadata']['length'][sample_id] - 2:
            # start equals end token
            span_end = span_start
            score_end = end_logits[sample_id][span_end]
        else:
            score_end, span_end = end_logits[sample_id][span_start:batch['metadata']['length'][sample_id] - 1].max(dim=0)
            span_end += span_start
        scores[sample_id] = (score_start + score_end).cpu().item()
        spans[sample_id] = (span_start, span_end)
        # extract span
        answers[sample_id] = context_wordpiece_tokens[sample_id][span_start:span_end + 1]

    return scores, spans, answers

import collections.abc


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def store_metrics(dir, filename, metrics, prefix = None):
    path = os.path.join(os.path.abspath(os.path.expanduser(dir)), filename + '.yaml')
    metrics = {prefix: metrics} if prefix is not None else {int(GLOBAL_SEED): metrics}
    
    try:
        with open(path, 'r') as metric_file:
            tmp = yaml.full_load(metric_file)
            update_dict(tmp, metrics)
            metrics = tmp
    except:
        pass

    # create necessary dirs
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path, 'w') as metric_file:
        yaml.dump(metrics, metric_file, indent=4, sort_keys=True)

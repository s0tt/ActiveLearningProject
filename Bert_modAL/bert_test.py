# delay evaluation of annotation
from __future__ import annotations

import sys 
import os
import re
import numpy as np

from typing import Dict, OrderedDict, Tuple, Union

import torch 
import random 

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from skorch import NeuralNetClassifier
from skorch import NeuralNet
from skorch.utils import to_numpy


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modAL'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Annotation_Interface'))

from modAL.dropout import mc_dropout_bald, mc_dropout_mean_st, mc_dropout_max_variationRatios, mc_dropout_max_entropy, _bald_divergence, _mean_standard_deviation, _entropy, _variation_ratios, set_dropout_mode
from modAL.models import DeepActiveLearner
from modAL.utils.selection import multi_argmax, shuffled_argmax
from modAL.utils.data import retrieve_rows

from transformers import BertModel

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from get_data_from_Bert import get_dataloader

from Labeling import label as getLabelStudioLabel
from Labeling import getLabelList



labels='single' # at the moment this is just set by hand ... 
device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_span(start_logits: torch.Tensor, end_logits: torch.Tensor, batch, softmax_applied: bool = True, topk: int = 1, extract_answer: bool = False, answer_only: bool = False):
    context_instance_start_end = batch['metadata']['context_instance_start_end']
    context = batch['metadata']['context']
    # token_to_context_idx = batch['metadata']['token_to_context_idx']
    context_wordpiece_tokens = batch['metadata']['cur_instance_context_tokens']
    # TODO use given lengths
    # lengths = torch.tensor([len(context) for context in context_wordpiece_tokens])
    # inputs are batch x sequence
    # TODO maybe vectorize computation of best span?
    num_samples = start_logits.size(0)
    scores = [None] * num_samples
    scores_all = [None] * num_samples
    spans = [None] * num_samples
    answers = [None] * num_samples
    max_score_start, max_spans_start = start_logits.max(dim=1)

    unpadded_probabilities = []
    for sample_id in range(num_samples):
        # consider all possible combinations (by addition) of start and end token
        # vectorize by broadcasting start/end token probabilites to matrix and adding both
        # afterward we can take the maximum of the upper half including the diagonal (end > start)
        slice_relevant_tokens = slice(len(batch['metadata']['question_tokens'][sample_id]) + 2, batch['metadata']['length'][sample_id] - 1)
        len_relevant_tokens = batch['metadata']['length'][sample_id] - 1 - len(batch['metadata']['question_tokens'][sample_id]) - 2
        
        #max_relevant_logits = start_logits[sample_id][slice_relevant_tokens].transpose()
        #print(max_relevant_logits)
        #unsqueezed_tokens = start_logits[sample_id][slice_relevant_tokens].unsqueeze(1)
        #print(unsqueezed_tokens)
        start_score_matrix = start_logits[sample_id][slice_relevant_tokens].expand(len_relevant_tokens, len_relevant_tokens)
        end_score_matrix = end_logits[sample_id][slice_relevant_tokens].expand(len_relevant_tokens, len_relevant_tokens) # new dimension is by default added to the front
        #out_matrix = start_score_matrix + end_score_matrix
        score_matrix = (start_score_matrix + end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0

        score_array = score_matrix[torch.triu(torch.ones(len_relevant_tokens, len_relevant_tokens)) == 1]

        # values can be lower than 0 -> make sure to set lower triangular matrix to very low value
        #lower_triangular_matrix = torch.tril(torch.ones_like(score_matrix, dtype=torch.long), diagonal=-1)
        #score_matrix.masked_fill_(lower_triangular_matrix, float("-inf")) # make sure that lower triangular matrix is set -inf to ensure end >= start

        probabilities = score_array.softmax(0)
        # TODO add maximum span length (mask diagonal)
        unpadded_probabilities.append(probabilities)

    # padding


    def padding_tensor(sequences):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask

    padded_tensors, masks = padding_tensor(unpadded_probabilities)

    return to_numpy(padded_tensors), np.array(to_numpy(masks), dtype=bool) # all scores as vector


def loss_function(output, target): 
    start_logits = output[0]
    end_logits = output[1]

    """
    we just use the single lable crossEntropyLoss

      if labels == 'single':
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1) # ignore out of context index
    else:
        loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    """

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1) # ignore out of context index

    # extract label
    # NOTE label contains offset for question + tokens
    if isinstance(loss_fn, (torch.nn.MultiLabelMarginLoss, torch.nn.MultiLabelSoftMarginLoss)):
        label_start, label_end = target.to(device).split(1, dim=1) # before: batch['label_multi'].to(self.device).split(1, dim=1) # all spans for this sample
    else:
        label_start, label_end = target.to(device).split(1, dim=1) # before batch['label'].to(self.device).split(1, dim=1) # one span

    # compute loss
    start_loss = loss_fn(start_logits, label_start.squeeze(1))


    end_loss = loss_fn(end_logits, label_end.squeeze(1))
    loss = start_loss + end_loss # TODO mean instead of sum?
    return loss 


class BertQA(torch.nn.Module):
    def __init__(self, cache_dir: Union[None, str] = None):
        super(BertQA, self).__init__()
        self.embedder = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.qa_outputs = torch.nn.Linear(self.embedder.config.hidden_size, 2, bias=True) # TODO include bias?
        self.soft_max = torch.nn.Softmax(dim=1)
        self.qa_outputs.apply(self.embedder._init_weights)


    def forward(self, inputs, segments, masks): 

        """
            I modified the input as well as the output of the forward function so that it can match with the input of my loss function and that it can accept the full batch

            token_ids: 
                The input ids are often the only required parameters to be passed to the model as input. 
                They are token indices, numerical representations of tokens building the sequences that will be used as input by the model.
                See: (https://huggingface.co/transformers/glossary.html)
            attention_mask: 
                The attention mask is an optional argument used when batching sequences together. 
                This argument indicates to the model which tokens should be attended to, and which should not.
                1 indicates a value that should be attended to, while 0 indicates a padded value.
                See: (https://huggingface.co/transformers/glossary.html)
            token_type_ids: 
                Some models’ purpose is to do sequence classification or question answering.
                These require two different sequences to be joined in a single “input_ids” entry, which usually is performed with the help of special tokens,
                such as the classifier ([CLS]) and separator ([SEP]) tokens.
                For example, the BERT model builds its two sequence input as such:
                The first sequence, the “context” used for the question, has all its tokens represented by a 0,
                whereas the second sequence, corresponding to the “question”, has all its tokens represented by a 1.
        """

        # input is batch x sequence
        # NOTE the order of the arguments changed from the pytorch pretrained bert package to the transformers package
        embedding, _ = self.embedder(inputs, token_type_ids=segments, attention_mask=masks)
        # only use context tokens for span prediction
        logits = self.qa_outputs(embedding)
        return logits


# Wrap pytorch class --> to give it an scikit-learn interface! 
classifier = NeuralNetClassifier(BertQA,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=AdamW,
                        train_split=None,
                        verbose=1,
                        device=device)


# initialize ActiveLearner

learnerBald = DeepActiveLearner(
    estimator=classifier, 
    criterion=torch.nn.NLLLoss,
    accept_different_dim=True,
    query_strategy=mc_dropout_bald
)

learnerMean = DeepActiveLearner(
    estimator=classifier, 
    criterion=torch.nn.NLLLoss,
    accept_different_dim=True,
    query_strategy=mc_dropout_mean_st
)


bert_qa = BertQA()
modules = list(bert_qa.modules()) # pick from here the Dopout indexes

"""
    At the moment the active learner requires that:  
    the dimensions of the new training data and label mustagree with the training data and labels provided so far

    But in our Bert-Model batch['input'] is never fixed --> we would need to adapt a bit modAL 


    Error: 
    File "/Library/Python/3.7/site-packages/modAL/utils/data.py", line 26, in data_vstack
        return np.concatenate(blocks)
    File "<__array_function__ internals>", line 6, in concatenate
        ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 209 and the array at index 1 has size 266

"""

data_loader = get_dataloader()
data_iter = iter(data_loader) # create iterator so that the same can be used in all function calls (also working with zip)

for batch in data_iter:

    inputs = batch['input']
    labels = batch['label']
    segments = batch['segments']
    masks = batch['mask']
    train_batch = {'inputs' : inputs, 'segments': segments, 'masks': masks}

    def Bert_training(batch, n_instances): 
        nr_instances = batch['input'].size()[0]

        probas = []
        metric_array = np.zeros((nr_instances))

        for i in range(2):

            set_dropout_mode(learnerBald.estimator.module_, [], train_mode=True)
            logits = learnerBald.estimator.infer(train_batch)
            start_logits, end_logits = logits.split(1, dim=-1)
            padded_tensors, mask = extract_span(start_logits, end_logits, batch, answer_only=True)
            probas.append(padded_tensors)

        metric_bald = _bald_divergence(probas, mask)
        metric_bald_unpadded = _bald_divergence(probas)

        metric_mean_std = _mean_standard_deviation(probas, mask)
        metric_mean_std_unpadded = _mean_standard_deviation(probas)

        metric_entropy = _entropy(probas, mask)
        metric_entropy_unpadded = _entropy(probas)

        metric_variation_ratios = _variation_ratios(probas, mask)
        metric_variation_ratios_unpadded = _variation_ratios(probas)

        """
            unpadded_probabilities = extract_span(start_logits, end_logits, batch, answer_only=True)
            for index, probability in enumerate(unpadded_probabilities): 
                probability = np.expand_dims(probability, axis=0)
                probas[index].append(probability)


        for index, proba in enumerate(probas): 
            metric_array[index] = _bald_divergence(proba)
        """

        max_indx, max_metrix = shuffled_argmax(metric_array, n_instances=2)

        inputs = retrieve_rows(batch['input'], max_indx)
        next_train_labels = retrieve_rows(batch['label'], max_indx)
        segments = retrieve_rows(batch['segments'], max_indx)
        masks = retrieve_rows(batch['mask'], max_indx)
        next_train_instances = {'inputs' : inputs, 'segments': segments, 'masks': masks}
        return next_train_instances, next_train_labels


    next_train_instances, next_train_labels = Bert_training(batch, 2)
    learnerBald.teach(X=next_train_instances, y=next_train_labels)

    """
    print(probas[0][-1])
    probas.flatten()
    print(np.max(probas))
    print(np.min(probas))

    print(np.argmax(probas))
    print(np.argmax(probas))

    bald_score = _bald_divergence(probas)

    print(bald_score)
    """


    learnerBald.teach(X=train_batch, y=labels)
    learnerMean.teach(X=train_batch, y=labels)

    print("Bald Learner:", learnerBald.score(train_batch, labels))
    print("Mean Learner:", learnerMean.score(train_batch, labels))
    

    print("Bald learner predict proba:", learnerBald.predict_proba(train_batch))
    print("Bald learner predict:", learnerBald.predict(train_batch))

    
    bald_idx, bald_instance, bald_metric = learnerBald.query(train_batch, n_instances=5, dropout_layer_indexes=[7, 16], num_cycles=10)
    mean_idx, mean_instance, mean_metric = learnerMean.query(train_batch, n_instances=4, dropout_layer_indexes=[7, 16], num_cycles=2)

    question = batch['metadata']['question']
    context = batch['metadata']['context']
    question_at_idx = question[bald_idx[0]]
    context_at_idx = context[bald_idx[0]]

    print("Send instance to label-studio... ")
    labelList = getLabelList(context, question, [bald_idx, mean_idx], [bald_metric, mean_metric], ["bald", "mean stddev"])
    label_queryIdx = getLabelStudioLabel(labelList)
    
    #learner.teach(X=special_input_array[mean_idx], y=labels[query_idx], only_new=False,)
    print("Question: ", question_at_idx)
    print("Oracle provided label:", label_queryIdx)

    break


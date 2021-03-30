# delay evaluation of annotation
from __future__ import annotations

import sys 
import os
import re
import numpy as np

from typing import Dict, OrderedDict, Tuple, Union

import torch 
import random 
import logging 


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


metric_name = sys.argv[1]

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'logs_BertQA_evaluation_{}.log'.format(metric_name)), level=logging.INFO)


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


if metric_name == 'bald': 
    query_strategy = mc_dropout_bald
elif metric_name == 'mean_std': 
    query_strategy = mc_dropout_mean_st
elif metric_name == 'max_variation': 
    query_strategy = mc_dropout_max_variationRatios
elif metric_name == 'max_entropy': 
    query_strategy = mc_dropout_max_entropy
elif metric_name == 'random': 
    query_strategy = mc_dropout_bald # just to pass something (will not be used)





n_initial = 0 # number of initial chosen samples for the training
num_model_training = 5
n_queries = 100
drawn_sampley_per_query = 10
forward_cycles_per_query = 50
output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)) , 'accuracies_{}.txt'.format(metric_name))


model_training_accuracies = []
x_axis = np.arange(n_initial, n_initial + n_queries*drawn_sampley_per_query + 1, drawn_sampley_per_query)
model_training_accuracies.append(x_axis)



bert_qa = BertQA()
modules = list(bert_qa.modules()) # pick from here the Dopout indexes

"""
    At the moment the active learner requires that:  
    the dimensions of the new training data and label mustagree with the training data and labels provided so far

    But in our Bert-Model batch['input'] is never fixed --> we would need to adapt a bit modAL 
"""

for idx_model_training in range(num_model_training): 

    learner = DeepActiveLearner(
        estimator=classifier, 
        query_strategy=query_strategy
    )


    learner.num_epochs = 2

    torch.cuda.manual_seed_all(idx_model_training)
    torch.manual_seed(idx_model_training)
    random.seed(idx_model_training)
    np.random.seed(idx_model_training)

    data_loader = get_dataloader()
    data_iter = iter(data_loader) # create iterator so that the same can be used in all function calls (also working with zip)


    # here we should do now the Pre-TRAINING

    for idx_query, batch in enumerate(data_iter):


        inputs = batch['input']
        labels = batch['label']
        segments = batch['segments']
        masks = batch['mask']
        train_batch = {'inputs' : inputs, 'segments': segments, 'masks': masks}

        def Bert_training(batch, n_instances=1, dropout_layer_indexes=[], num_cycles=10): 
            nr_instances = batch['input'].size()[0]

            probas = []

            set_dropout_mode(learner.estimator.module_, dropout_layer_indexes, train_mode=True)

            for i in range(num_cycles):

                X.detach()
        
                probas = []
                for X_split in torch.split(X, sample_per_forward_pass):

                logits = learner.estimator.infer(train_batch)
                start_logits, end_logits = logits.split(1, dim=-1)
                padded_tensors, mask = extract_span(start_logits, end_logits, batch, answer_only=True)
                probas.append(padded_tensors)

            metrics = query_strategy(probas, mask)

            max_indx, max_metric = shuffled_argmax(metrics, n_instances=n_instances)

            inputs = retrieve_rows(batch['input'], max_indx)
            next_train_labels = retrieve_rows(batch['label'], max_indx)
            segments = retrieve_rows(batch['segments'], max_indx)
            masks = retrieve_rows(batch['mask'], max_indx)
            next_train_instances = {'inputs' : inputs, 'segments': segments, 'masks': masks}
            return next_train_instances, next_train_labels

        next_train_instances, next_train_labels = Bert_training(batch, n_instances=4, dropout_layer_indexes=[7, 16], num_cycles=10)
        learner.teach(X=next_train_instances, y=next_train_labels)

        print("Bald Learner:", learner.score(train_batch, labels))



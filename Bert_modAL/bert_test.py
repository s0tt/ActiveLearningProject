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


metric_name = 'random'#sys.argv[1]

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'logs_BertQA_evaluation_{}.log'.format(metric_name)), level=logging.INFO)


labels='single' # at the moment this is just set by hand ... 
device = "cuda" if torch.cuda.is_available() else "cpu"


def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    # from: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''Calculate F1 score. Can work with gpu tensors
        
        The original implmentation is written by Michal Haltuf on Kaggle.
        
        Returns
        -------
        torch.Tensor
            `ndim` == 1. 0 <= val <= 1
        
        Reference
        ---------
        - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
        
        '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return f1

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

    return out_tensor, mask # to_numpy(out_tensor), np.array(to_numpy(mask), dtype=bool)

def extract_span(start_logits: torch.Tensor, end_logits: torch.Tensor, batch, maximilian: bool = True, softmax_applied: bool = True, topk: int = 1, extract_answer: bool = False, answer_only: bool = False):
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

        # Maximilians realisation 

        start_score_matrix = start_logits[sample_id][0][slice_relevant_tokens].unsqueeze(1).expand(len_relevant_tokens, len_relevant_tokens).double()
        end_score_matrix = end_logits[sample_id][0][slice_relevant_tokens].unsqueeze(1).transpose(0, 1).expand(len_relevant_tokens, len_relevant_tokens).double() # new dimension is by default added to the front

        if maximilian: 
            score_matrix = (start_score_matrix + end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0
        else: 
            score_matrix = (start_score_matrix*end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0
        
        # my realisation


        score_array = score_matrix[torch.triu(torch.ones(len_relevant_tokens, len_relevant_tokens)) == 1]
        
        # values can be lower than 0 -> make sure to set lower triangular matrix to very low value
        #lower_triangular_matrix = torch.tril(torch.ones_like(score_matrix, dtype=torch.long), diagonal=-1)
        #score_matrix.masked_fill_(lower_triangular_matrix, float("-inf")) # make sure that lower triangular matrix is set -inf to ensure end >= start
        
        if softmax_applied: 
            probabilities = score_array.softmax(0)
        else: 
            probabilities = score_array
        # TODO add maximum span length (mask diagonal)
        unpadded_probabilities.append(probabilities)

    # padding

    return unpadded_probabilities

def calculate_f1_score_Bert(test_set, learner):
    # label part
    labels = test_set['label_multi']
    start_labels, end_labels = labels.split(1, dim=1)
    unpadded_labels = extract_span(start_labels, end_labels, test_set, softmax_applied=False, maximilian=False, answer_only=True) # this gives us the prediction
    padded_label, masks = padding_tensor(unpadded_labels)

    # prediction part
    logits_predictions = learner.estimator.module_(test_set['input'], test_set['segments'], test_set['mask']) # test if it uses the setted batch size ... 
    start_logits, end_logits = logits_predictions.transpose(1, 2).split(1, dim=1)
    unpadded_predictions = extract_span(start_logits, end_logits, test_set, softmax_applied=True, maximilian=False, answer_only=True)
    padded_predictions, masks = padding_tensor(unpadded_predictions)

    overall_f1_loss = 0
    for instance_label, instance_prediction in zip(padded_label, padded_predictions): 
        overall_f1_loss += f1_loss(instance_label, instance_prediction)

    return overall_f1_loss

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
    learner.batch_size = 2

    torch.cuda.manual_seed_all(idx_model_training)
    torch.manual_seed(idx_model_training)
    random.seed(idx_model_training)
    np.random.seed(idx_model_training)

    data_loader = get_dataloader()
    data_iter = iter(data_loader) # create iterator so that the same can be used in all function calls (also working with zip)


    # here we should do now the Pre-TRAINING


    f1_scores = []
    f1_score = calculate_f1_score_Bert(batch, learner) 
    f1_score.append(f1_score)
    logging.info("Metric name: {}, model training run: {}, initial f1_score: {}".format(metric_name, idx_model_training, f1_score))

    for idx_query, batch in enumerate(data_iter):


        def get_next_train_instances(batch, n_instances=1, dropout_layer_indexes=[], num_cycles=10, sample_per_forward_pass=5): 
            nr_instances = batch['input'].size()[0]

            batch['input'].detach()
            batch['segments'].detach()
            batch['masks'].detach()

            predictions = []

            set_dropout_mode(learner.estimator.module_, dropout_layer_indexes, train_mode=True)

            for i in range(num_cycles):
        
                probas = []
                for inputs, segments, masks in zip(torch.split(batch['input'], sample_per_forward_pass), torch.split(batch['segments'], sample_per_forward_pass), torch.split(batch['masks'], sample_per_forward_pass)): 

                    logits = learner.estimator.infer(inputs, segments, masks})
                    start_logits, end_logits = logits.transpose(1, 2).split(1, dim=-1)
                    unpadded_probabilities = extract_span(start_logits, end_logits, batch, softmax_applied=True, maximilian=False, answer_only=True)
                    
                    probas += unpadded_probabilities
                
                padded_tensors, masks = padding_tensor(probas)
                predictions.append(padded_tensors)

            metrics = query_strategy(predictions, mask)

            max_indx, max_metric = shuffled_argmax(metrics, n_instances=n_instances)

            inputs = retrieve_rows(batch['input'], max_indx)
            next_train_labels = retrieve_rows(batch['label'], max_indx)
            segments = retrieve_rows(batch['segments'], max_indx)
            masks = retrieve_rows(batch['mask'], max_indx)
            next_train_instances = {'input' : inputs, 'segments': segments, 'mask': masks}
            return next_train_instances, next_train_labels

        next_train_instances, next_train_labels = get_next_train_instances(batch, n_instances=4, dropout_layer_indexes=[7, 16], num_cycles=10, sample_per_forward_pass=5)
        learner.teach(X=next_train_instances, y=next_train_labels)



        f1_score = calculate_f1_score_Bert(batch, learner)

        f1_scores.append(f1_score) 
        logging.info("Metric name: {}, model training run: {}, query number: {}, f1_score: {}".format(metric_name, idx_model_training, idx_query, f1_scores))

    model_training_accuracies.append(np.array(accuracies).T)


logging.info("Result: {}".format(model_training_accuracies))
np.savetxt(output_file, np.array(model_training_accuracies).T, delimiter=' ')

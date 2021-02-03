# delay evaluation of annotation
from __future__ import annotations

import sys 
import os
import re
import numpy as np

from typing import Dict, OrderedDict, Tuple, Union

import torch 
from skorch import NeuralNetClassifier
from skorch import NeuralNet

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modAL'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Annotation_Interface'))

from modAL.dropout import mc_dropout
from modAL.models import DeepActiveLearner
from transformers import BertModel

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from get_data_from_Bert import get_dataloader

from Labeling import label as getLabelStudioLabel

labels='single' # at the moment this is just set by hand ... 
device = "cuda" if torch.cuda.is_available() else "cpu"


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


    def forward(self, input): 
        
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

        mask_tensor = input[:, 2]
        segment_tensor = input[:, 1] 
        token_ids = input[:, 0]  

        # input is batch x sequence
        # NOTE the order of the arguments changed from the pytorch pretrained bert package to the transformers package
        embedding, _ = self.embedder(token_ids, token_type_ids=segment_tensor, attention_mask=mask_tensor)
        # only use context tokens for span prediction
        logits = self.qa_outputs(embedding)
        probabilities = self.soft_max(logits)
        return probabilities


# Wrap pytorch class --> to give it an scikit-learn interface! 
classifier = NeuralNetClassifier(BertQA,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=AdamW,
                        train_split=None,
                        verbose=1,
                        device=device)


# initialize ActiveLearner

learner = DeepActiveLearner(
    estimator=classifier, 
    criterion=torch.nn.NLLLoss,
    accept_different_dim=True,
    query_strategy=mc_dropout
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

    
    inputs = torch.Tensor.cpu(inputs).detach().numpy()
    labels = torch.Tensor.cpu(labels).detach().numpy()
    segments = torch.Tensor.cpu(segments).detach().numpy()
    masks = torch.Tensor.cpu(masks).detach().numpy()

    special_input_array = np.array([])
    
    i = 0
    for (input, label, segment, mask) in zip(inputs, labels, segments, masks):
        if i == 0: 
            special_input_array = np.array([[input, segment, mask]])
        else: 
            one_row = np.array([[input, segment, mask]])
            special_input_array = np.append(special_input_array, one_row, axis=0)
        i +=1 


    special_input_array = torch.from_numpy(special_input_array)
    labels = torch.from_numpy(labels)


    learner.teach(X=special_input_array, y=labels)

    print(learner.score(special_input_array, labels))
 
    
    query_idx, query_instance, metric = learner.query(special_input_array, n_instances=1, dropout_layer_indexes=[7, 16], num_cycles=10)
  
    question = batch['metadata']['question']
    context = batch['metadata']['context']
    question_at_idx = question[query_idx[0]]
    context_at_idx = context[query_idx[0]]
    print("Send instance to label-studio... ")
    label_queryIdx = getLabelStudioLabel(question_at_idx, context_at_idx)
    
    #learner.teach(X=special_input_array[query_idx], y=labels[query_idx], only_new=False,)
    print("Question: ", question_at_idx)
    print("Oracle provided label:", label_queryIdx)

    break


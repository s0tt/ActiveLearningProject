# delay evaluation of annotation
from __future__ import annotations

import os
import re
import numpy as np

from typing import Dict, OrderedDict, Tuple, Union

import torch 
from skorch import NeuralNetClassifier
from skorch import NeuralNet
from modAL.models import ActiveLearner
from transformers import BertModel

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from get_data_from_Bert import y_pool, input_batch

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

    print(start_logits)
    print(label_start)
    print(label_end)
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

    def forward(self, token_ids): # return type also modified

        """
            I modified the input as well as the output of the forward function so that it can match with the input of my loss function and that it can accept the full batch
        """

        segment_ids = input_batch['segments'].to(device)
        lengths = input_batch['mask'].to(device)
        # input is batch x sequence
        # NOTE the order of the arguments changed from the pytorch pretrained bert package to the transformers package
        embedding, _ = self.embedder(token_ids, token_type_ids=segment_ids, attention_mask=lengths)
        # only use context tokens for span prediction
        logits = self.qa_outputs(embedding)
        # split last dim to get separate vectors for start and end
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        # set padded values in output to small value # TODO consider the same for question + tokens?
        mask = (1-lengths).bool()

        start_logits.masked_fill_(mask, -1e7)

        end_logits.masked_fill_(mask, -1e7)
        
        # modification of the output 
        #output = [start_logits, end_logits]

        print("one forward pass finished")
        print(start_logits)
        return start_logits

# initialize loss (when do we have to use what loss .... )

classifier = NeuralNet(BertQA,
                                 criterion=torch.nn.CrossEntropyLoss,
                                 optimizer=AdamW,
                                 train_split=None,
                                 verbose=1,
                                 device=device)

#print(classifier.fit(input_batch, labels))



# initialize ActiveLearner


input_data = input_batch['input'].to(device)

learner = ActiveLearner(
    estimator=classifier, 
    X_training=input_data, y_training=y_pool, 
)

"""
learner.teach(
    X=input_batch, y=labels, only_new=False,
)

print(learner.score(input_batch, labels))
"""
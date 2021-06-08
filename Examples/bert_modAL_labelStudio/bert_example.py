import sys 
import os
import numpy as np

from typing import Union

import torch 
from skorch import NeuralNetClassifier

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../Annotation_Interface'))

from modAL.dropout import mc_dropout_bald
from modAL.models import DeepActiveLearner
from transformers import BertModel

from transformers import AdamW

from data_preprocessing.dataloader import get_dataloader

from LabelingClass import LabelInstance
from Labeling import getLabelList

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 5


def extract_span(logits: torch.Tensor, batch, addition_instead_of_multiplication: bool = False, softmax_applied: bool = False, get_label:bool=False):

    """
        Logits adapter funktion for BERT-QA. 
        In general a logits adaptor function is needed, in the learner.query method, when the output of a forward pass is not a trivial 
        logit vector as in multi class classification tasks. 
    """

    if get_label:
        start_logits, end_logits = logits.split(1, dim=1)
    else:
        start_logits, end_logits = logits.transpose(1, 2).split(1, dim=1)
    
    num_samples = start_logits.size(0)

    unified_len = round((len(batch['inputs'][0]) * (len(batch['inputs'][0]) + 1))/2)  #Gaussian sum formula for sequences
    if get_label:
        unpadded_probabilities = torch.empty(size=(num_samples, 1))
    else:
        unpadded_probabilities = torch.empty(size=(num_samples, unified_len))
    for sample_id in range(num_samples):
        # consider all possible combinations (by addition) of start and end token
        # vectorize by broadcasting start/end token probabilites to matrix and adding both
        # afterward we can take the maximum of the upper half including the diagonal (end > start)
        
        mask = batch['masks'][sample_id]
        nr_mask = np.sum(mask.numpy()) #sum mask to get nr of total valid tokens
        nr_segments = np.sum(batch['segments'][sample_id][mask == 1].numpy()) #sum masked segments to get nr of answer tokens

        start_idx = nr_mask-nr_segments
        end_idx = nr_mask-1-1 #index is mask nr-1 and one more -1 for excluding [SEP]
        len_relevant_tokens = end_idx-start_idx

        score_matrix_pad = torch.full(size=(len(mask), len(mask)), fill_value=float("nan"))

        start_score_matrix = start_logits[sample_id][0][start_idx:end_idx].unsqueeze(1).expand(len_relevant_tokens, len_relevant_tokens).double()
        end_score_matrix = end_logits[sample_id][0][start_idx:end_idx].unsqueeze(1).transpose(0, 1).expand(len_relevant_tokens, len_relevant_tokens).double() # new dimension is by default added to the front

        if addition_instead_of_multiplication: 
            score_matrix = (start_score_matrix + end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0
        else: 
            score_matrix = (start_score_matrix*end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0

        score_matrix_pad[start_idx:end_idx,start_idx:end_idx] = score_matrix
        score_array = score_matrix_pad[torch.triu(torch.ones_like(score_matrix_pad) == 1)]
        
        # values can be lower than 0 -> make sure to set lower triangular matrix to very low value

        if softmax_applied: 
            score_array[~score_array.isnan()] = score_array[~score_array.isnan()].softmax(0)
            probabilities = score_array
        else: 
            probabilities = score_array

        if get_label:
            probabilities[probabilities.isnan()] = -1 #set to -1 for argmax to work correctly
            unpadded_probabilities[sample_id, 0] = torch.argmax(probabilities)
        else:
            unpadded_probabilities[sample_id, :] = probabilities

    # padding
    return unpadded_probabilities

class BertQA(torch.nn.Module):
    def __init__(self, cache_dir: Union[None, str] = None):
        super(BertQA, self).__init__()
        self.embedder = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.qa_outputs = torch.nn.Linear(self.embedder.config.hidden_size, 2, bias=True) # TODO include bias?
        self.soft_max = torch.nn.Softmax(dim=1)
        self.qa_outputs.apply(self.embedder._init_weights)


    def forward(self, inputs, segments, masks): 

        """
            inputs: 
                The input ids are often the only required parameters to be passed to the model as input. 
                They are token indices, numerical representations of tokens building the sequences that will be used as input by the model.
                See: (https://huggingface.co/transformers/glossary.html)
            masks: 
                The attention mask is an optional argument used when batching sequences together. 
                This argument indicates to the model which tokens should be attended to, and which should not.
                1 indicates a value that should be attended to, while 0 indicates a padded value.
                See: (https://huggingface.co/transformers/glossary.html)
            segments: 
                Some models’ purpose is to do sequence classification or question answering.
                These require two different sequences to be joined in a single “input_ids” entry, which usually is performed with the help of special tokens,
                such as the classifier ([CLS]) and separator ([SEP]) tokens.
                For example, the BERT model builds its two sequence input as such:
                The first sequence, the “context” used for the question, has all its tokens represented by a 0,
                whereas the second sequence, corresponding to the “question”, has all its tokens represented by a 1.
        """
        embedding, _ = self.embedder(inputs, token_type_ids=segments, attention_mask=masks)
        # only use context tokens for span prediction
        logits = self.qa_outputs(embedding)

        return logits


##### Initialize modAL DeepActiveLearner with Skorch classifier, loss, query strategy ####
classifier = NeuralNetClassifier(BertQA,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=AdamW,
                        train_split=None,
                        verbose=1,
                        device=device,
                        max_epochs=1)


##### Initialize modAL DeepActiveLearner with Skorch classifier, loss, query strategy ####
learner = DeepActiveLearner(
    estimator=classifier, 
    criterion=torch.nn.NLLLoss,
    accept_different_dim=True,
    query_strategy=mc_dropout_bald
)

bert_qa = BertQA()
modules = list(bert_qa.modules()) # pick from here the Dopout indexes


##### Initialize label studio instance ####
labelSystem = LabelInstance(80, {'text': 'Text', 'question': 'Text'}, 
                    '[["accuracy", "percent"]]',
                    '{"bald": "Divergence factor represents uncertainty. Lower is better."}'
                    )

##### Load and process modell training data ####
data_loader = get_dataloader(batch_size)
data_iter = iter(data_loader) # create iterator so that the same can be used in all function calls (also working with zip)


##### Start the active learning cycle loop ####
for batch in data_iter:

    inputs = batch['input']
    labels = batch['label']
    segments = batch['segments']
    masks = batch['mask']
    
    train_batch = {'inputs' : inputs, 'segments': segments, 'masks': masks}
    labels = batch['label']
    
    learner.teach(X=train_batch, y=labels)

    #print("Bald Learner:", learnerBald.score(train_batch, labels))     #leaner.score does not work with multi-dimensional BERT model outputs
    print("Bald learner predict proba:", learner.predict_proba(train_batch))
    print("Bald learner predict:", learner.predict(train_batch))

    #calculates defined query strategy on given data samples
    bald_idx, bald_instance, bald_metric = learner.query(train_batch, n_instances=2, dropout_layer_indexes=[207, 213], num_cycles=10, logits_adaptor=extract_span)

    question = batch['metadata']['question']
    context = batch['metadata']['context']
    question_at_idx = [question[int(idx)] for idx in bald_idx]

    print("Send instance to label-studio... ")

    #score = learner.score(test_batch, labels) #exemplary calculation of model score only with test data here
    score = np.array([0]) #dummy value as leaner.score does not work with multi-dimensional model outputs
    
    labelList = getLabelList(context, question, [bald_idx], [bald_metric], ["bald"])
    oracle_responses = labelSystem.label(labelList, [score])
    
    
    new_train_idxs = []
    new_labels = torch.empty((len(oracle_responses), 2), dtype=torch.int64)
    for idx, response in enumerate(oracle_responses):

        #match oracle provided responses to batch data
        sample_idx = question.index(response["data"]["question"])
        new_train_idxs.append(int(sample_idx))

        #collect new data labels which where answered by oracle
        #TODO: One would have to map label studio charSpans (textwise) to input token indices (token-wise) to  provide correct labels
        #TODO: Currently just text indices are used which does not match token indices --> might lead to IndexError out of bound
        new_labels[idx, :] = torch.from_numpy(np.array(response["charSpans"][0]).astype(int))

        print("Question: ", question[sample_idx])
        print("Oracle provided label:", response["charSpans"])

    #assemble new train set from user labeled data
    train_at_idx = {'inputs' : inputs[new_train_idxs], 'segments': segments[new_train_idxs], 'masks': masks[new_train_idxs]}

    #train the model further with newly acquired data labels
    learner.teach(X=train_at_idx, y=new_labels, warm_start=True)


# delay evaluation of annotation
from __future__ import annotations

import sys 
import os
import re
import numpy as np
from typing import Dict, OrderedDict, Tuple, Union
import time
from collections import Counter

import torch 
import random 
import logging 
import argparse

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

from transformers import BertTokenizer, get_linear_schedule_with_warmup

from get_data_from_Bert import get_dataloader

from Labeling import label as getLabelStudioLabel
from Labeling import getLabelList

from sklearn.metrics import f1_score



parser = argparse.ArgumentParser(description='BertQA-argparse')
parser.add_argument('-m','--metric-name', help='Which metric should be used', type=str, required=True)
parser.add_argument('-i','--initial-samples', help='Number of initial samples', type=int, required=True)
parser.add_argument('-ip','--initial-pool-size', help='Number of initial samples in the pool', type=int, required=True)
parser.add_argument('-nq','--number-of-queries', help='Number of queries for one active learning cycle', type=int, required=True)
parser.add_argument('-fc','--forward-cycles-per-query', help='Number of forward cycles per query', type=int, required=True)
parser.add_argument('-ds','--drawn-samples-per-query', help='Number of drawn-samples-per-query', type=int, required=True)
parser.add_argument('-bs','--batch-size', help='Batch size', type=int, required=True)
parser.add_argument('-ts','--test-size', help='Test batch size', type=int, required=True)


args = vars(parser.parse_args())

metric_name = args['metric_name']

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'logs_BertQA_evaluation_{}_i_{}_ip_{}_nq_{}_fc_{}_ds_{}_bs_{}.log'.format(metric_name, args['initial_samples'], args['initial_pool_size'], args['number_of_queries'], args['forward_cycles_per_query'], args['drawn_samples_per_query'], args['batch_size'])), filemode='w', level=logging.INFO)


labels='single' # at the moment this is just set by hand ... 
device = "cuda" if torch.cuda.is_available() else "cpu"

cuda = torch.device('cuda') 


def f1(prediction, truth):
    prediction_tokens = Counter(prediction)
    truth_tokens = Counter(truth)
    num_overlaps = sum((prediction_tokens & truth_tokens).values())
    if num_overlaps > 0:
        # f1 score is bigger than 0
        precision = num_overlaps / sum(prediction_tokens.values())
        recall = num_overlaps / sum(truth_tokens.values())
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0.0


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
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    """
    
    # vectorised version
    tp = (y_true * y_pred).sum(dim=1).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=1).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=1).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=1).to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1_score = f1.sum().item()

    """
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1_score = f1.item()
    """


    return f1_score

def extract_span(start_logits: torch.Tensor, end_logits: torch.Tensor, batch, maximilian: bool = True, softmax_applied: bool = False, topk: int = 1, extract_answer: bool = False, answer_only: bool = False, get_label:bool=False):

    num_samples = start_logits.size(0)

    unified_len = round((batch['input'][0].shape[0] * (batch['input'][0].shape[0] + 1))/2)  # round((len(batch['input'][0]) * (len(batch['input'][0]) + 1))/2)  #Gaussian sum formula for sequences
    if get_label:
        unpadded_probabilities = torch.empty(size=(num_samples, 1), device=cuda)
    else:
        unpadded_probabilities = torch.empty(size=(num_samples, unified_len), device=cuda)
    for sample_id in range(num_samples):
        # consider all possible combinations (by addition) of start and end token
        # vectorize by broadcasting start/end token probabilites to matrix and adding both
        # afterward we can take the maximum of the upper half including the diagonal (end > start)
        

        #TODO: [CLS] Hwhere is the city? [SEP] the city is there
        mask = batch['mask'][sample_id]
        #out_matrix = torch.tensor([1, float('nan'), 2])
        #out_matrix = np.nan(shape=(len(mask), len(mask)))
        


        nr_mask = mask.sum() #sum mask to get nr of total valid tokens
        nr_segments = batch['segments'][sample_id][mask == 1].sum() #sum masked segments to get nr of answer tokens

        start_idx = nr_mask-nr_segments
        end_idx = nr_mask-1-1 #index is mask nr-1 and one more -1 for excluding [SEP]
        len_relevant_tokens = end_idx-start_idx

        #slice_relevant_tokens = slice(len(batch['metadata']['question_tokens'][sample_id]) + 2, batch['metadata']['length'][sample_id] - 1)
        #len_relevant_tokens = batch['metadata']['length'][sample_id] - 1 - len(batch['metadata']['question_tokens'][sample_id]) - 2
        
        #max_relevant_logits = start_logits[sample_id][slice_relevant_tokens].transpose()
        #print(max_relevant_logits)
        #unsqueezed_tokens = start_logits[sample_id][slice_relevant_tokens].unsqueeze(1)
        #print(unsqueezed_tokens)

        # Maximilians realisation 
        #start_score_matrix= np.nan(shape=(len(mask), len(mask)))
        #end_score_matrix= np.nan(shape=(len(mask), len(mask)))
       # start_score_matrix_pad = torch.full(size=(len(mask), len(mask)), fill_value=float("nan"))
        #end_score_matrix_pad = torch.full(size=(len(mask), len(mask)), fill_value=float("nan"))

        score_matrix_pad = torch.full(size=(mask.shape[0], mask.shape[0]), fill_value=float("nan"), device=cuda)

        #vec_start = start_logits[sample_id][0][slice_relevant_tokens].unsqueeze(1).double()
        #start_score_matrix[slice_relevant_tokens][slice_relevant_tokens] = torch.matmul()

        #start_score_matrix = start_logits[sample_id][0][slice_relevant_tokens].unsqueeze(1).double()
        #end_score_matrix = end_logits[sample_id][0][slice_relevant_tokens].unsqueeze(1).transpose(0, 1)

        start_score_matrix = start_logits[sample_id][0][start_idx:end_idx].unsqueeze(1).expand(len_relevant_tokens, len_relevant_tokens).double()
        end_score_matrix = end_logits[sample_id][0][start_idx:end_idx].unsqueeze(1).transpose(0, 1).expand(len_relevant_tokens, len_relevant_tokens).double() # new dimension is by default added to the front
        #start_score_matrix_pad[start_idx:end_idx,start_idx:end_idx] = start_score_matrix[1:-1, 1:-1]
        #end_score_matrix_pad[start_idx:end_idx,start_idx:end_idx] = end_score_matrix[1:-1, 1:-1]

        if maximilian: 
            score_matrix = (start_score_matrix + end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0
            #score_matrix = (start_score_matrix_pad + end_score_matrix_pad).triu() # return upper triangular part including diagonal, rest is 0
        else: 
            score_matrix = (start_score_matrix*end_score_matrix).triu() # return upper triangular part including diagonal, rest is 0
            #score_matrix = (start_score_matrix_pad*end_score_matrix_pad).triu() # return upper triangular part including diagonal, rest is 0
        
        # my realisation
        

        #score_array = score_matrix[torch.triu(torch.ones_like(score_matrix)) == 1]
        score_matrix_pad[start_idx:end_idx,start_idx:end_idx] = score_matrix
        score_array = score_matrix_pad[torch.triu(torch.ones_like(score_matrix_pad) == 1)]
        
        # values can be lower than 0 -> make sure to set lower triangular matrix to very low value
        #lower_triangular_matrix = torch.tril(torch.ones_like(score_matrix, dtype=torch.long), diagonal=-1)
        #score_matrix.masked_fill_(lower_triangular_matrix, float("-inf")) # make sure that lower triangular matrix is set -inf to ensure end >= start
        
        if softmax_applied: 
            score_array[~score_array.isnan()] = score_array[~score_array.isnan()].softmax(0)
            probabilities = score_array
        else: 
            probabilities = score_array
        # TODO add maximum span length (mask diagonal)
        if get_label:
            probabilities[probabilities.isnan()] = -1 #set to -1 for argmax to work correctly
            unpadded_probabilities[sample_id, 0] = torch.argmax(probabilities)
        else:
            unpadded_probabilities[sample_id, :] = probabilities

    # padding
    return unpadded_probabilities

def extract_span_v_2(logits: torch.Tensor, batch):

    topk = 1
    softmax_applied=False # is done in the get_predictions
    maximilian=False
    answer_only=True
    extract_answer = False
    answer_only = False
    get_label = False

    start_logits, end_logits = logits.transpose(1, 2).split(1, dim=1)

    return extract_span(start_logits, end_logits, batch, softmax_applied=False, maximilian=False, answer_only=True)

def calculate_f1_score_Bert(test_set, learner):

    overall_f1_loss=0

    """
    start_label, end_label = test_set['label_multi'].split(1, dim=1)
    argmax_index_start = start_label[0].argmax()
    argmax_index_end = end_label[0].argmax()
    len_question = len(test_set['metadata']['question_tokens'][0])

    answer_self_extracted = test_set['metadata']['context_tokens'][0][argmax_index_start-len_question-2:argmax_index_end-len_question-1]

    answer = test_set['metadata']['original_answers'][0][0][0].lower().split()
    """

    with torch.no_grad():
        # f1 score calculation for Question-Answering
        logits = learner.estimator.forward({'input' : test_set['input'], 'segments' : test_set['segments'], 'mask': test_set['mask']}) 
        
        
        start_logits, end_logits = logits.transpose(1, 2).split(1, dim=1)
        
        start_predicted_classes = start_logits.argmax(dim=2).numpy().flatten()
        end_predicted_classes = end_logits.argmax(dim=2).numpy().flatten()


        start_label, end_label = test_set['label_multi'].split(1, dim=1)
        
        #start_classes = start_label.argmax(dim=2).numpy().flatten()
        #end_classes = end_logits.argmax(dim=2).numpy().flatten()

        for index, (start_prediction, end_predcition) in enumerate(zip(start_predicted_classes, end_predicted_classes)):
            len_question = len(test_set['metadata']['question_tokens'][index])

            if (start_prediction-len_question-2 <0): 
                start = 0
                logging.info("Still totally invalid predictions")
            else: 
                start = start_prediction-len_question-2
            prediction = test_set['metadata']['context_tokens'][index][start:end_predcition-len_question-1]
            truth = test_set['metadata']['original_answers'][index][0][0].lower().split()
            overall_f1_loss += f1(prediction, truth)

        """
        #f1 score calculation only direct match
        logits = learner.estimator.forward({'input' : test_set['input'], 'segments' : test_set['segments'], 'mask': test_set['mask']}) 
        
        
        start_logits, end_logits = logits.transpose(1, 2).split(1, dim=1)
        
        start_predicted_classes = start_logits.argmax(dim=2).numpy().flatten()
        end_predicted_classes = end_logits.argmax(dim=2).numpy().flatten()

        start_label, end_label = test_set['label_multi'].split(1, dim=1)

        start_classes = start_label.argmax(dim=2).numpy().flatten()
        end_classes = end_logits.argmax(dim=2).numpy().flatten()

        overall_f1_loss_start = f1_score(start_classes, start_predicted_classes, average='micro')
        overall_f1_loss_end = f1_score(end_classes, end_predicted_classes, average='micro')
         (overall_f1_loss_end + overall_f1_loss_start)/2 
        """
    return overall_f1_loss

class BertQA(torch.nn.Module):
    def __init__(self, cache_dir: Union[None, str] = None):
        super(BertQA, self).__init__()
        self.embedder = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.qa_outputs = torch.nn.Linear(self.embedder.config.hidden_size, 2, bias=True) # TODO include bias?
        self.qa_outputs.apply(self.embedder._init_weights)


    def forward(self, input, segments, mask):         
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
        embedding, _ = self.embedder(input, token_type_ids=segments, attention_mask=mask)
        # only use context tokens for span prediction
        logits = self.qa_outputs(embedding)
        return logits


logging.info("GPU _allocation before classifier: {}".format(torch.cuda.memory_allocated()))

# Wrap pytorch class --> to give it an scikit-learn interface! 
classifier = NeuralNetClassifier(BertQA,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=torch.optim.Adam,
                        train_split=None,
                        verbose=1,
                        lr=3e-05, 
                        device=device)

logging.info("GPU _allocation after classifier: {}".format(torch.cuda.memory_allocated()))




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



n_initial = args['initial_samples'] #2 # number of initial chosen samples for the training
num_model_training = 5
n_queries = args['number_of_queries']
drawn_samples_per_query = args['drawn_samples_per_query']
forward_cycles_per_query = args['forward_cycles_per_query']
sample_per_forward_pass = args['batch_size'] # same as batch size
output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)) , 'f1_scores_{}.txt'.format(metric_name))

model_training_f1_scores = []
x_axis = np.arange(n_initial, n_initial + n_queries*drawn_samples_per_query + 1, drawn_samples_per_query)
model_training_f1_scores.append(x_axis)


train_dataset = 'SQuAD-train'
batch_size_train_dataloader = args['initial_pool_size']+n_initial#86588### 
test_dataset = 'SQuAD-dev'  
batch_size_test_dataloader = args['test_size'] #10507


logging.info("GPU _allocation: {}".format(torch.cuda.memory_allocated()))


# get test batch
data_loader_test = get_dataloader([test_dataset], batch_size_test_dataloader)
data_iter_test = iter(data_loader_test) 
test_batch = 0

for batch in data_iter_test: 
    test_batch = batch
    break

# label part
#labels_f1_score = extract_span(start_logits, end_logits, test_batch, softmax_applied=False, maximilian=False, answer_only=True)
#del labels_f1_score # just to test if this tensor does still consume memory


logging.info("GPU _allocation: {}".format(torch.cuda.memory_allocated()))


for idx_model_training in range(num_model_training): 

    # initialise learner and set seeds
    learner = DeepActiveLearner(
        estimator=classifier, 
        query_strategy=query_strategy
    )

    learner.num_epochs = 2
    learner.batch_size = sample_per_forward_pass

    torch.cuda.manual_seed_all(idx_model_training)
    torch.manual_seed(idx_model_training)
    random.seed(idx_model_training)
    np.random.seed(idx_model_training)

    # gets for us the train data (shuffle --> so that the data is always new sorted)

    data_loader_train = get_dataloader([train_dataset], batch_size_train_dataloader, shuffle=True)
    data_iter_train = iter(data_loader_train) 

    for batch in data_iter_train: 
        train_data = batch
        break



    # assemble initial data & pool data 
    initial_idx = np.random.choice(range(len(train_data['input'])), size=n_initial, replace=False)
    X_initial = {'input': train_data['input'][initial_idx], 'segments': train_data['segments'][initial_idx], 'mask': train_data['mask'][initial_idx]}
    y_initial = train_data['label'][initial_idx]

    pool_initial = {'input': np.delete(train_data['input'], initial_idx, axis=0), 
                      'segments': np.delete(train_data['segments'], initial_idx, axis=0), 
                      'mask': np.delete(train_data['mask'], initial_idx, axis=0)
                    }
    pool_labels = np.delete(train_data['label'], initial_idx, axis=0)
    
    del train_data

    logging.info("Pool size x {}".format(pool_initial['input'].size()))
    logging.info("Initial size x {}".format(X_initial['input'].size()))
    logging.info("GPU _allocation before teach: {}".format(torch.cuda.memory_allocated()))

    
    # here we should do now the Pre-TRAINING
    learner.teach(X=X_initial, y=y_initial)
    del X_initial, y_initial

    logging.info("GPU _allocation after teach: {}".format(torch.cuda.memory_allocated()))

    f1_scores = []
    
    f1_ = calculate_f1_score_Bert(test_batch, learner) 
    f1_scores.append(f1_)
    logging.info("Metric name: {}, model training run: {}, initial f1_score: {}".format(metric_name, idx_model_training, f1_))
    
    pool = pool_initial
    del pool_initial


    for idx_query in range(n_queries):
        
        start_query = time.time()
        query_idx, query_instance, query_metric = learner.query(pool, n_instances=drawn_samples_per_query, dropout_layer_indexes=[207, 213], num_cycles=forward_cycles_per_query, sample_per_forward_pass=sample_per_forward_pass)
        logging.info("Time for a single query: {}".format(time.time()-start_query))


        if metric_name == 'random': 
            query_idx = np.random.choice(range(len(pool['input'])), size=drawn_samples_per_query, replace=False)

        next_train_instances = {'input': pool['input'][query_idx], 
                      'segments': pool['segments'][query_idx], 
                      'mask': pool['mask'][query_idx] 
                    }

        learner.teach(X=next_train_instances, y=pool_labels[query_idx])

        pool = {'input': np.delete(pool['input'], query_idx, axis=0), 
                    'segments': np.delete(pool['segments'], query_idx, axis=0), 
                    'mask': np.delete(pool['mask'], query_idx, axis=0)
                }

        pool_labels = np.delete(pool_labels, query_idx, axis=0)

        f1_ = calculate_f1_score_Bert(test_batch, learner)
        f1_scores.append(f1_) 
        logging.info("Metric name: {}, model training run: {}, query number: {}, f1_score: {}".format(metric_name, idx_model_training, idx_query, f1_))

    model_training_f1_scores.append(np.array(f1_scores).T)


logging.info("Result: {}".format(model_training_f1_scores))
np.savetxt(output_file, np.array(model_training_f1_scores).T, delimiter=' ')

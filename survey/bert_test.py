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

from modAL.dropout import mc_dropout_bald, mc_dropout_mean_st, mc_dropout_max_variationRatios, mc_dropout_max_entropy, mc_dropout_multi
from modAL.models import DeepActiveLearner
from transformers import BertModel

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from get_data_from_Bert import get_dataloader

from Labeling import getLabelList
from LabelingClass import LabelInstance

import pickle
import json
import difflib
import time
from datetime import datetime
import shutil

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


def extract_span(start_logits: torch.Tensor, end_logits: torch.Tensor, batch, maximilian: bool = True, softmax_applied: bool = False, topk: int = 1, extract_answer: bool = False, answer_only: bool = False, get_label:bool=False):
    num_samples = start_logits.size(0)
    scores = [None] * num_samples
    scores_all = [None] * num_samples
    spans = [None] * num_samples
    answers = [None] * num_samples
    max_score_start, max_spans_start = start_logits.max(dim=1)
    



    unified_len = round((len(batch['input'][0]) * (len(batch['input'][0]) + 1))/2)  #Gaussian sum formula for sequences
    if get_label:
        unpadded_probabilities = torch.empty(size=(num_samples, 1))
    else:
        unpadded_probabilities = torch.empty(size=(num_samples, unified_len))
    for sample_id in range(num_samples):
        # consider all possible combinations (by addition) of start and end token
        # vectorize by broadcasting start/end token probabilites to matrix and adding both
        # afterward we can take the maximum of the upper half including the diagonal (end > start)
        

        #TODO: [CLS] Hwhere is the city? [SEP] the city is there
        mask = batch['mask'][sample_id]
        #out_matrix = torch.tensor([1, float('nan'), 2])
        #out_matrix = np.nan(shape=(len(mask), len(mask)))
        


        nr_mask = np.sum(mask.numpy()) #sum mask to get nr of total valid tokens
        nr_segments = np.sum(batch['segments'][sample_id][mask == 1].numpy()) #sum masked segments to get nr of answer tokens

        slice_relevant_tokens = np.arange(nr_mask-nr_segments,nr_mask)
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

        score_matrix_pad = torch.full(size=(len(mask), len(mask)), fill_value=float("nan"))

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
            unpadded_probabilities[sample_id] = torch.argmax(probabilities)
        else:
            unpadded_probabilities[sample_id, :] = probabilities

    # padding
    if get_label:
        unpadded_probabilities = unpadded_probabilities.squeeze().type(torch.long)
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

        ##### NEW Change output of BERT QA to 1dim ######

        start_logits, end_logits = logits.transpose(1, 2).split(1, dim=1)
        batch = {'input' : inputs, 'segments': segments, 'mask': masks}
        unpadded_probabilities = extract_span(start_logits, end_logits, batch, softmax_applied=True, maximilian=False, answer_only=True)
        #padded_tensors, masks = padding_tensor(unpadded_probabilities)
        return unpadded_probabilities



####restart the whole application
def restart_program(labelSystem, timeStamp):
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    print("Restarting the whole program....")

    #remove all user files
    time.sleep(2)
    source_dir = "./questionAnswering/completions/"
    if os.path.exists(source_dir):
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), "./userResults/"+timeStamp+"/completions/")

    time.sleep(2)
    labelSystem.stopServer()

    time.sleep(2)
    folder = './questionAnswering'
    file_list = os.listdir(folder)
    for filename in file_list:
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    print("Removed {} files from {}".format(str(len(file_list)) ,folder))
    time.sleep(5)
    #python = sys.executable
    #os.execl(python, python, * sys.argv)
    #os.execv(sys.argv[0], sys.argv)
    #raise Exception("Kill script")

def exportUserResults(timeStamp, run, results, statistics, startTime=None):
    if not os.path.isdir("./userResults"):
        os.makedirs("./userResults")

    if not os.path.isdir("./userResults/"+timeStamp):
        os.makedirs("./userResults/"+timeStamp)
    
    if not os.path.isdir("./userResults/"+timeStamp+"/completions"):
        os.makedirs("./userResults/"+timeStamp+"/completions")

    f_name = "./userResults/"+timeStamp+"/"+timeStamp+".json"
    if not os.path.isfile(f_name):
        resDict = {}
    else:
        with open(f_name, "r") as json_file:
            resDict = json.load(json_file)
    
    dateTimeObj = datetime.now()
    timeStr = dateTimeObj.strftime("%m-%d-%Y_%H:%M:%S")
    if run > 0:
        resDict[run] = {}
        resDict[run]["time"] = timeStr
        resDict[run]["mean_bald"] = statistics[0]
        resDict[run]["accuracy"] = statistics[1]
        for idx,result in enumerate(results):
            resDict[run][idx] = {}
            resDict[run][idx]["data"] = result["data"]
            resDict[run][idx]["charSpans"] = result["charSpans"]
            resDict[run][idx]["texts"] = result["texts"]
        if startTime is not None:
            resDict["startClickTime"] = startTime

    with open(f_name, "w") as json_file:
        json.dump(resDict, json_file, indent=4)

def survey():
    dateTimeObj = datetime.now()
    timeStr = dateTimeObj.strftime("%m-%d-%Y-%H-%M-%S")
    exportUserResults(timeStr, 0, None, None) #create files/folders

    #Label Studio instance
    print("###################Starting survey###################")
    labelSystem = LabelInstance(80, {'text': 'Text', 'question': 'Text'}, 
                    '[["mean_bald_uncertainty", "bald_score"], ["accuracy", "percent"]]',
                    '{"bald": "Divergence facotor represents uncertainty. Lower is better.", "mean stddev": "Mean standard deviation between sample runs. Lower is better","max entropy": "Maximum entropy between sample runs. Lower is better","max variation": "Maximium variation between sample runs. Lower is better"}'
                    )


    # # Wrap pytorch class --> to give it an scikit-learn interface! 
    # classifier = NeuralNetClassifier(BertQA,
    #                         criterion=torch.nn.CrossEntropyLoss,
    #                         optimizer=AdamW,
    #                         train_split=None,
    #                         verbose=1,
    #                         device=device,
    #                         max_epochs=1)

    # # initialize ActiveLearner

    # learner = DeepActiveLearner(
    #     estimator=classifier, 
    #     criterion=torch.nn.NLLLoss,
    #     accept_different_dim=True,
    #     query_strategy=mc_dropout_multi
    # )

    # bert_qa = BertQA()
    # modules = list(bert_qa.modules()) # pick from here the Dopout indexes

    #["bald", "mean stddev", "max entropy", "max variation"]

    # data_loader = get_dataloader()
    # data_iter = iter(data_loader) # create iterator so that the same can be used in all function calls (also working with zip)

    #pickle.dump(batch, open("batch_survey.pkl", "wb"))

    batch = pickle.load(open("batch_survey.pkl", "rb"))


    # for batch in data_iter:

    inputs = batch['input']
    labels = batch['label']
    segments = batch['segments']
    masks = batch['mask']
    train_batch = {'inputs' : inputs, 'segments': segments, 'masks': masks}

    # # label part
    # labels = batch['label_multi']
    # start_labels, end_labels = labels.split(1, dim=1)
    # labels = extract_span(start_labels, end_labels, batch, softmax_applied=False, maximilian=False, answer_only=True, get_label=True) # this gives us the prediction


    # learnerBald.teach(X=train_batch, y=labels)
    # learnerMean.teach(X=train_batch, y=labels)

    # print("Bald Learner:", learnerBald.score(train_batch, labels))
    # print("Mean Learner:", learnerMean.score(train_batch, labels))


    # print("Bald learner predict proba:", learnerBald.predict_proba(train_batch))
    # print("Bald learner predict:", learnerBald.predict(train_batch))



    #bald_idx, bald_instance, bald_metric = learnerBald.query(train_batch, n_instances=5, dropout_layer_indexes=[7, 16], num_cycles=10)
    #multi_idx, multi_instance, multi_metric = learner.query(train_batch, n_instances=4, dropout_layer_indexes=[7, 16], num_cycles=2)
    multi_instance = pickle.load(open("multi_instance.pkl", "rb"))
    multi_metric = pickle.load(open("multi_metric.pkl", "rb"))


    print("Send instance to label-studio... ")

    nr_instances = 50
    nr_label_cycles = 1
    mean_step = 0.05 #%
    acc_step = 0.005 # %

    def evalLabels(responses, labelList, statistics):
        for response in responses:
            sample_idx = question.index(response["data"]["question"])
            correct_label_num = batch["label"][sample_idx]
            correct_label_txt = batch["metadata"]["answers_per_instance"][sample_idx][0]

            #correct labeled
            matching_score = difflib.SequenceMatcher(a=response["texts"][0].lower(), b=correct_label_txt.lower()).ratio()
            if matching_score > 0.90: #label is fully or nearly correct
                statistics[0] -= np.abs((matching_score * (mean_step*statistics[0])))
                statistics[1] += (matching_score * acc_step)
            else: #label not correct
                statistics[0] += np.abs((matching_score * (mean_step*statistics[0])))
                statistics[1] -= (matching_score * acc_step)

            #remove already labeled samples from list
            list_idx = None
            questionList = [question[1] for question in labelList]
            try:
                list_idx = questionList.index(response["data"]["question"])
            except:
                list_idx=None
            
            if list_idx is not None:
                labelList.pop(list_idx)

        return statistics

    context = batch["metadata"]["context"][0:nr_instances]
    question = batch["metadata"]["question"][0:nr_instances]
    idx = np.arange(0,nr_instances)


    labelList = getLabelList(context, question, [idx, idx, idx, idx], [multi_metric["bald"][0:nr_instances], multi_metric["mean_st"][0:nr_instances], multi_metric["max_entropy"][0:nr_instances], multi_metric["max_var"][0:nr_instances]], 
                                ["bald", "mean stddev", "max entropy", "max variation"])


    statistics = [np.mean(multi_metric["bald"]), 0.8] #Mean Bald score, Model Accuracy




    for i in range(nr_label_cycles):
        print("Survey Cycle: " + str(i+1))
        labelRes, startTime = labelSystem.label(labelList, statistics)
        statistics = evalLabels(labelRes, labelList, statistics)
        exportUserResults(timeStr, i+1, labelRes, statistics, startTime if i==0 else None)

    time.sleep(2)
    labelSystem.endSurvey()
    labelSystem.label(labelList, statistics, getLabels=False) #call label last time to perform user redirect

    #learner.teach(X=special_input_array[mean_idx], y=labels[query_idx], only_new=False,)
    # print("Question: ", question_at_idx)
    # print("Oracle provided label:", label_queryIdx)
    return labelSystem, timeStr

if __name__ == '__main__':
    while True:
        labelSystem, timeStamp = survey()
        restart_program(labelSystem, timeStamp)
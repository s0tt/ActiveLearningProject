# delay evaluation of annotation
from __future__ import annotations

import sys 
import os
import re
import numpy as np

from typing import Dict, OrderedDict, Tuple, Union

# import torch 
# from skorch import NeuralNetClassifier
# from skorch import NeuralNet

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','modAL'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','Annotation_Interface'))

# from modAL.dropout import mc_dropout_bald, mc_dropout_mean_st, mc_dropout_max_variationRatios, mc_dropout_max_entropy, mc_dropout_multi
# from modAL.models import DeepActiveLearner
# from transformers import BertModel

# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor
# from torchvision.datasets import MNIST

# from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

# from get_data_from_Bert import get_dataloader

from Labeling import getLabelList
from LabelingClass import LabelInstance

import pickle
import json
import difflib
import time
from datetime import datetime
import shutil

####restart the whole application
def restart_program(labelSystem, timeStamp):
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    print("Restarting the whole program....")



    time.sleep(2)
    labelSystem.stopServer()
    print("Label System stop send....")
    if labelSystem.thread_id is not None:
        print("Force terminate thread....")
        labelSystem.thread_id.terminate()

    #remove all user files
    time.sleep(2)
    source_dir = os.path.join("questionAnswering", "completions", "")
    if os.path.exists(source_dir):
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), os.path.join("userResults", timeStamp,"completions",""))

    print("Moved completion files....")
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
    time.sleep(2)
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

    batch = pickle.load(open("batch_survey_py.pkl", "rb"))


    # for batch in data_iter:

    #inputs = batch['input']
    #labels = batch['label']
    #segments = batch['segments']
    #masks = batch['mask']
    #train_batch = {'inputs' : inputs, 'segments': segments, 'masks': masks}

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
    #multi_instance = pickle.load(open("multi_instance_py.pkl", "rb"))
    multi_metric = pickle.load(open("multi_metric_py.pkl", "rb"))


    print("Send instance to label-studio... ")

    nr_instances = 50
    nr_label_cycles = 3
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
                statistics[0] += np.abs(((1-matching_score) * (mean_step*statistics[0])))
                statistics[1] -= ((1-matching_score) * acc_step)

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
import sys
import os
import torch
import random
import logging 

from torch import nn
from skorch import NeuralNetClassifier
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modAL'))

from modAL.models import DeepActiveLearner, ActiveLearner
from modAL.dropout import mc_dropout_bald, mc_dropout_mean_st, mc_dropout_max_variationRatios, mc_dropout_max_entropy
from modAL.uncertainty import margin_sampling

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

metric_name = sys.argv[1]

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'logs_mnist_evaluation_{}.log'.format(metric_name)), level=logging.INFO)


torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# build class for the skorch API
class Torch_Model(nn.Module):
    def __init__(self,):
        super(Torch_Model, self).__init__()
        self.convs = nn.Sequential(
                                nn.Conv2d(1,32,3),
                                nn.ReLU(),
                                nn.Conv2d(32,64,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.25)
        )
        self.fcs = nn.Sequential(
                                nn.Linear(12*12*64,128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128,10),
        )

    def forward(self, x):
        out = x
        out = self.convs(out)
        out = out.view(-1,12*12*64)
        out = self.fcs(out)

        return out

torch_model = Torch_Model()

layer_list = list(torch_model.modules())

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = NeuralNetClassifier(Torch_Model,
                                 criterion=torch.nn.CrossEntropyLoss,
                                 optimizer=torch.optim.Adam,
                                 train_split=None,
                                 verbose=1,
                                 device=device, 
                                 lr=0.001)


mnist_data = MNIST('.', download=True, transform=ToTensor())
dataloader = DataLoader(mnist_data, shuffle=False, batch_size=60000)
X_train, y_train = next(iter(dataloader))
X_train = X_train.reshape(60000, 1, 28, 28)


mnist_data = MNIST('.', train=False, download=True, transform=ToTensor())
dataloader = DataLoader(mnist_data, shuffle=False, batch_size=10000)
X_test, y_test = next(iter(dataloader))
X_test = X_test.reshape(10000, 1, 28, 28)



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


for idx_model_training in range(num_model_training): 

    # initialize ActiveLearner
    learner = DeepActiveLearner(
        estimator=classifier, 
        query_strategy=query_strategy,  
    )

    learner.num_epochs = 10

    torch.cuda.manual_seed_all(idx_model_training)
    torch.manual_seed(idx_model_training)
    random.seed(idx_model_training)
    np.random.seed(idx_model_training)


    X_pool_initial = X_train
    y_pool_initial = y_train

    logging.info("Pool size x {}".format(X_pool_initial.size()))
    logging.info("Test size x {}".format(y_pool_initial.size()))
    logging.info("Initial size x: 0")


    # the active learning loop
    X_teach = []
    y_teach = []
    X_pool = X_pool_initial
    y_pool = y_pool_initial
    accuracies = []

    accuracy = learner.estimator.score(X_test, y_test)
    accuracies.append(accuracy)
    logging.info("Metric name: {}, model training run: {}, initial accuracy: {}".format(metric_name, idx_model_training, accuracy))

    for idx_query in range(n_queries):
        print('Query no. %d' % (idx_query + 1))

        if metric_name != 'random': 
            query_idx, query_instance, metric = learner.query(X_pool, n_instances=drawn_sampley_per_query, num_cycles=forward_cycles_per_query, 
                                                            sample_per_forward_pass=100)
        else: 
            query_idx, query_instance, metric = learner.query(X_pool, n_instances=drawn_sampley_per_query, num_cycles=forward_cycles_per_query, 
                                                            sample_per_forward_pass=100)
            query_idx = np.random.choice(range(len(X_pool)), size=drawn_sampley_per_query, replace=False)
            print(query_idx.shape)

        # Add queried instances
        if type(X_teach) == list: 
            X_teach = X_pool[query_idx]
            y_teach = y_pool[query_idx]
        else: 
            X_teach  = torch.cat((X_teach, X_pool[query_idx]))
            y_teach  = torch.cat((y_teach, y_pool[query_idx]))

        learner.teach(X_teach, y_teach, warm_start=False)

        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # scoring part
        accuracy = learner.estimator.score(X_test, y_test)
        accuracies.append(accuracy) 
        logging.info("Metric name: {}, model training run: {}, query number: {}, accuracy: {}".format(metric_name, idx_model_training, idx_query, accuracy))

    model_training_accuracies.append(np.array(accuracies).T)


logging.info("Result: {}".format(model_training_accuracies))
np.savetxt(output_file, np.array(model_training_accuracies).T, delimiter=' ')


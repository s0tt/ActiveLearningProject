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

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.realpath(__file__)),'logs_mnist_evaluation_{}.log'.format(metric_name))    , level=logging.INFO)

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

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
dataloader = DataLoader(mnist_data, shuffle=True, batch_size=60000)
X_train, y_train = next(iter(dataloader))
X_train = X_train.reshape(60000, 1, 28, 28)


mnist_data = MNIST('.', train=False, download=True, transform=ToTensor())
dataloader = DataLoader(mnist_data, shuffle=True, batch_size=10000)
X_test, y_test = next(iter(dataloader))
X_test = X_test.reshape(10000, 1, 28, 28)

# assemble initial data
n_initial = 250
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]


# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)


if metric_name == 'bald': 
    query_strategy = mc_dropout_bald
elif metric_name == 'mean_std': 
    query_strategy = mc_dropout_mean_st
elif metric_name == 'max_variation': 
    query_strategy = mc_dropout_max_variationRatios
elif metric_name == 'mean_entropy': 
    query_strategy = mc_dropout_max_entropy
elif metric_name == 'random': 
    query_strategy = mc_dropout_bald # just to pass something (will not be used)



# initialize ActiveLearner
learner = DeepActiveLearner(
    estimator=classifier, 
    query_strategy=query_strategy,  
)

logging.info("Pool size x {}".format(X_pool.size()))
logging.info("Test size x {}".format(X_test.size()))
logging.info("Initial size x {}".format(X_initial.size()))

learner.num_epochs = 10
num_model_training = 5
n_queries = 100
forward_cycles_per_query = 50
output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)) , 'accuracies_{}.txt'.format(metric_name))


model_training_accuracies = []
x_axis = np.arange(1000, 2001, 10)
model_training_accuracies.append(x_axis)


for idx_model_training in range(num_model_training): 


    learner.teach(X_initial, y_initial, warm_start=False) # Initial teaching --> resets parameters

    # the active learning loop
    X_teach = X_initial
    y_teach = y_initial
    accuracies = []

    accuracy = learner.estimator.score(X_test, y_test)
    accuracies.append(accuracy)
    logging.info("Metric name: {}, model training run: {}, initial accuracy: {}".format(metric_name, idx_model_training, accuracy))

    for idx_query in range(n_queries):
        print('Query no. %d' % (idx_query + 1))

        if metric_name != 'random': 
            query_idx, query_instance, metric = learner.query(X_pool, n_instances=10, num_cycles=forward_cycles_per_query)
        else: 
            query_idx = np.random.choice(range(len(X_pool)), size=10, replace=False)

        # Add queried instances
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


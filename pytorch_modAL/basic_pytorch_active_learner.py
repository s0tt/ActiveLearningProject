import sys
import os
import torch
from torch import nn
from skorch import NeuralNetClassifier
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modAL'))

from modAL.models import DeepActiveLearner, ActiveLearner
from modAL.dropout import mc_dropout_bald, mc_dropout_mean_st
from modAL.uncertainty import margin_sampling

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

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
                                 criterion=torch.nn.CrossEntropyLoss,# torch.nn.NLLLoss,
                                 optimizer=torch.optim.Adam,
                                 train_split=None,
                                 verbose=1,
                                 device=device)


mnist_data = MNIST('.', download=True, transform=ToTensor())
dataloader = DataLoader(mnist_data, shuffle=True, batch_size=60000)
X, y = next(iter(dataloader))

# read training data
X_train, X_test, y_train, y_test = X[:50000], X[50000:], y[:50000], y[50000:]
X_train = X_train.reshape(50000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

# assemble initial data
n_initial = 1000
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]


# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)[:5000]
y_pool = np.delete(y_train, initial_idx, axis=0)[:5000]


# initialize ActiveLearner
learner = DeepActiveLearner(
    estimator=classifier, 
    query_strategy=mc_dropout_mean_st,  
)
learner.teach(X_initial, y_initial) # not necessary anymore now

print("Score from sklearn: {}".format(learner.estimator.score(X_pool, y_pool)))
print("Model prediction: {}".format(learner.predict_proba(X_test)))


# the active learning loop
n_queries = 10
X_teach = X_initial
y_teach = y_initial


for idx in range(n_queries):
    print('Query no. %d' % (idx + 1))
    query_idx, query_instance, metric = learner.query(X_pool, n_instances=100, num_cycles=5)
    # Add queried instances
    X_teach  = torch.cat((X_teach, X_pool[query_idx]))
    y_teach  = torch.cat((y_teach, y_pool[query_idx]))
    learner.teach(X_teach, y_teach)

    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    print("Model score: {}".format(learner.score(X_test, y_test))) # give us the model performance 
    print("Model prediction: {}".format(learner.predict_proba(X_test)))
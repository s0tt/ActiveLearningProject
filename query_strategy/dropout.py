import numpy as np
import logging

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def mc_dropout(classifier: BaseEstimator, X, nr_cycles : int = 50,
                         **mc_dropout_kwargs) -> np.ndarray:
    """
    MVP Query strategy

    """ 

    # set dropout layers to train mode
    if not classifier.estimator.initialized_:
        classifier.estimator.initialize()
        logging.getLogger().info("Dropout: Initialized classifier manually")

    set_dropout_mode(classifier.estimator.module_, train_mode=True)

    #iterate for each sample
    pred_list = []
    #for idx, sample in enumerate(X):
        #run nr_cycles NN inference
    predictions = []
    for i in range(nr_cycles):
        logging.getLogger().info("Dropout: start prediction forward pass")
        prediction = classifier.estimator.predict_proba(X)
        predictions.append(prediction)
    pred_list.append(predictions)
    

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_, train_mode=False)
    return pred_list


def set_dropout_mode(model, train_mode: bool):
    """ Function to enable the dropout layers by setting them to train mode """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if True == train_mode:
                m.train()
                logging.getLogger().info("Dropout: set mode of " + str(m.__class__.__name__) + " to train")
            elif False == train_mode:
                m.eval()
                logging.getLogger().info("Dropout: set mode of " + str(m.__class__.__name__) + " to eval")

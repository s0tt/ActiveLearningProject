import numpy as np
import logging

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def mc_dropout(classifier: BaseEstimator, X, num_cycles : int = 50,
                         **mc_dropout_kwargs) -> np.ndarray:
    """
    MVP Query strategy

    """ 

    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_, train_mode=True)

    predictions = []

    #for each batch run num_cycles forward passes
    for i in range(num_cycles):
        logging.getLogger().info("Dropout: start prediction forward pass")
        #call Skorch infer function to perform model forward pass
        #In comparison to: predict(), predict_proba() the infer() 
        # does not change train/eval mode of other layers 
        prediction = classifier.estimator.infer(X)
        predictions.append(prediction)

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_, train_mode=False)

    #TODO: implement querye selection measure (e.g. BALD (Bayesian active learning divergence))

    #TODO: format selected query as modAL tuple to fix current data format issues

    #to inspect MC Dropout forward passes you can set a breakpoint here
    return pred_list


def set_dropout_mode(model, train_mode: bool):
    """ Function to enable the dropout layers by setting them to user specified mode (bool: train_mode)"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if True == train_mode:
                m.train()
                logging.getLogger().info("Dropout: set mode of " + str(m.__class__.__name__) + " to train")
            elif False == train_mode:
                m.eval()
                logging.getLogger().info("Dropout: set mode of " + str(m.__class__.__name__) + " to eval")

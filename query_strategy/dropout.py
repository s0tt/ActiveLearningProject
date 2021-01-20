import numpy as np
import logging

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.uncertainty import _proba_entropy
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def mc_dropout(classifier: BaseEstimator, X, n_instances: int = 1, dropout_layer_indexes: list = [], 
                num_cycles : int = 50, **mc_dropout_kwargs) -> np.ndarray:
    """
    MVP Query strategy
    TODO: Add n_instances support
    """ 

    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=True)

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
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=False)

    #TODO: implement querye selection measure (e.g. BALD (Bayesian active learning divergence))
    bald = _bald_divergence(predictions)

    #TODO: format selected query as modAL tuple to fix current data format issues

    #to inspect MC Dropout forward passes you can set a breakpoint here
    return predictions

def _bald_divergence(proba) -> np.ndarray:
    accumulated_score = np.zeros(shape=proba[0].shape)
    accumulated_entropy = np.zeros(shape=(proba[0].shape[0], ))

    for dropout_cycle in proba:
        accumulated_score += proba[dropout_cycle]
        accumulated_entropy += _proba_entropy(proba[dropout_cycle])
    
    # average inter score entropy
    f_x = accumulated_entropy / len(proba)
    
    # average entropy in scores
    average_score = accumulated_score / len(proba)
    g_x = _proba_entropy(average_score)
    
    u_x = g_x - f_x
    return u_x

def set_dropout_mode(model, dropout_layer_indexes: list, train_mode: bool):
    """ 
        Function to enable the dropout layers by setting them to user specified mode (bool: train_mode)
        TODO: Reduce maybe complexity
        TODO: Keras support
    """

    modules = list(model.modules()) # list of all modules in the network.
    
    if len(dropout_layer_indexes) != 0:  
        for index in dropout_layer_indexes: 
            layer = modules[index]
            if layer.__class__.__name__.startswith('Dropout'): 
                if True == train_mode:
                    layer.train()
                elif False == train_mode:
                    layer.eval()
            else: 
                raise KeyError("The passed index: {} is not a Dropout layer".format(index))

    else: 
        for module in modules:
            if module.__class__.__name__.startswith('Dropout'):
                if True == train_mode:
                    module.train()
                    logging.getLogger().info("Dropout: set mode of " + str(module.__class__.__name__) + " to train")
                elif False == train_mode:
                    module.eval()
                    logging.getLogger().info("Dropout: set mode of " + str(module.__class__.__name__) + " to eval")

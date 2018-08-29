"""Functions to carry out different types of evaluation on the models.

negative_log_likelihood:
  Compute negative log-likelihood of the probability distribution predicted by
  a model, given the target values for certain input.

accuracy:
  Compute prediction accuracy of the predictions of a model.
"""


import numpy as np


def negative_log_likelihood(probs, tgts):
    """
    Compute negative log-likelihood of the probability distribution predicted
    by a model, given the target values for certain input.

    Input:
    ------
    probs: The predicted distributions for each sample of input data.
    tgts: Target values corresponding to each input.

    Output:
    -------
    Cross entropy.
    """
    print type(tgts)
    print type(np.arange(tgts.shape[0]))
    print(tgts.dtype)
    if tgts.dtype!=np.int32:
        tgts=tgts.astype(np.int32)
    return -np.mean(np.log(probs)[np.arange(tgts.shape[0]), tgts])


def accuracy(y_pred, y_test):
    """
    Compute the accuracy of predictions made by a model given the target
    values.

    Input:
    ------
    y_pred: The predictions for each sample of input data.
    tgts: Target values corresponding to each input.

    Output:
    -------
    Accuracy.
    """
    return np.float(np.sum(y_pred == y_test)) / np.shape(y_pred)[0]

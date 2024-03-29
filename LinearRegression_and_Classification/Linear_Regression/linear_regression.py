"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    X = np.nan_to_num(X)
    w = np.nan_to_num(w)
    y = np.nan_to_num(y)
    err = np.nanmean(np.square(np.dot(X, w) - y))
    return err


###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################
    w = None
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    X_t = np.transpose(X)
    right_part = np.dot(X_t, y)
    left_part = np.dot(X_t, X)
    w = np.dot(np.linalg.inv(left_part), right_part)
    return w


###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    I = np.identity(X.shape[1])
    X_t = np.transpose(X)
    right_part = np.dot(X_t, y)
    left_part = np.dot(X_t, X)
    e,v = np.linalg.eig(left_part)
    temp_min = np.min(np.abs(e))
    while temp_min < (10**-5):
        left_part = left_part.__add__(0.1 * I)
        e, v = np.linalg.eig(left_part)
        temp_min = np.min(np.abs(e))
    w = np.dot(np.linalg.inv(left_part), right_part)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here #
    #####################################################
    w = None
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    I = np.identity(X.shape[1])
    X_t = np.transpose(X)
    right_part = np.dot(X_t, y)
    left_part = np.dot(X_t, X)
    # e, v = np.linalg.eig(left_part)
    # temp_min = np.min(np.abs(e))
    # while temp_min < (10 ** -5):
    #     left_part = left_part.__add__(lambd * I)
    #     e, v = np.linalg.eig(left_part)
    #     temp_min = np.min(np.abs(e))
    left_part = left_part.__add__(lambd * I)
    w = np.dot(np.linalg.inv(left_part), right_part)
    return w


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    min_error = float("inf")
    for i in range(-19, 20):
        weights = regularized_linear_regression(Xtrain, ytrain, 10 ** i)
        curr_error = mean_square_error(weights, Xval, yval)
        if min_error > curr_error:
            min_error = curr_error
            bestlambda = 10 ** i
    return bestlambda

    # max_val = (float("inf"), float("inf"))
    # for i in range(-19, 20):
    #     err = mean_square_error(regularized_linear_regression(Xtrain, ytrain, 10 ** i), Xval, yval)
    #     if err < max_val[0]:
    #         max_val = (err, 10 ** i)
    # return max_val[1]


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    # test_X = np.asarray([[1,2,3],[0,0,5]])
    # X = test_X
    # power = 4

    if power == 1:
        return X
    X_mapped = X
    for i in range(2, power+1):
        X_mapped = np.concatenate((X_mapped, np.power(X, i)), axis=1)
    return X_mapped

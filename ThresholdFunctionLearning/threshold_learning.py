##########################################################################################
# Filename: threshold_learning.py
#
# Purpose: to Learn a threshold function and get binary label predictions 
#          for multi-label classifiers
#
# Author(s): Bobby (Robert) Lumpkin
# 
# Library Dependencies: numpy, sklearn
###########################################################################################
 
import numpy as np
from sklearn.linear_model import LinearRegression
 
## Define a function to compute size of 'mistake sets' given the predicted values for a training instance 'Y_pred', 
## the true labels for that instance 'Y' and a threshold value 't'
def mistake_size(Y_train_pred, Y_train, t):
    ## Define the 'false_negatives' set
    positive_labels_bool = Y_train == 1
    positive_labels_pred_vals = Y_train_pred[positive_labels_bool]
    num_false_negatives = sum(positive_labels_pred_vals <= t)
    negative_labels_bool = Y_train == 0
    negative_labels_pred_vals = Y_train_pred[negative_labels_bool]
    num_false_positives = sum(negative_labels_pred_vals >= t)
    mistake_size = num_false_positives + num_false_negatives
    return mistake_size
 
## Define a function to get the target value, 't(x)', for a single observation, x, i.e. argmin_t(mistake_size)
def target_value_single(Y_train_pred, Y_train, t_range):
    t_vals = np.linspace(t_range[0], t_range[1], 100)
    mistake_size_array = len(Y_train_pred) * 2 * np.ones((1, 100))
    i = 0
   
    # Get mistake set size for each value of t
    for t in t_vals:
        mistake_size_array[0, i] = mistake_size(Y_train_pred, Y_train, t)
        i = i + 1
        
    # Get the t with smallest mistake set
    argmin_index = np.argmin(mistake_size_array)
    argmin_val = t_vals[argmin_index]
    return argmin_val
 
## Define a function to compute the target values for entire training dataset
def get_target_values(Y_train_pred, Y_train, t_range):
    # Initialize 'target' vector
    target_values_full = t_range[1] * 2 * np.ones((1, len(Y_train)))
   
    # Update 'target_values_full' with calculated targets
    for i in range(len(Y_train_pred)):
        Y_train_pred_single = Y_train_pred[i]
        Y_train_single = Y_train[i]
        target_value_i = target_value_single(Y_train_pred_single, Y_train_single, t_range)
        target_values_full[0, i] = target_value_i
       
    return target_values_full   
 
## Define a function to learn threshold function parameters
def threshold_function_learn(Y_train_pred, Y_train, t_range):
    # Get the target values
    target_values = get_target_values(Y_train_pred, Y_train, t_range)
   
    # Fit a linear regression model using 'Y_pred' as covariates and 'target_values' as targets
    fit = LinearRegression().fit(Y_train_pred, target_values.transpose())
   
    return fit
 
## Define a function to get binary test label predictions, given a learned threshold function fit and test propensity predictions.
def predict_labels_binary(Y_pred, threshold_function):
    # Get the threshold values
    threshold_vals = threshold_function.predict(Y_pred)
    Y_pred_binary = 2 * np.ones((Y_pred.shape[0], Y_pred.shape[1]))
    for i in range(Y_pred.shape[0]):
        for j in range(Y_pred.shape[1]):
            if Y_pred[i, j] > threshold_vals[i]:
                Y_pred_binary[i, j] = 1
            else:
                Y_pred_binary[i, j] = 0
               
    return Y_pred_binary

## Define a function to learn threshold function from training labels and return binary test label predictions and the threshold function
def predict_test_labels_binary(Y_train_pred, Y_train, Y_test_pred, t_range):
    # Get the learned threshold function
    fit = threshold_function_learn(Y_train_pred, Y_train, t_range)
    threshold_vals = fit.predict(Y_test_pred)
    Y_test_pred_binary = 2 * np.ones((Y_test_pred.shape[0], Y_test_pred.shape[1]))
    for i in range(Y_test_pred.shape[0]):
        for j in range(Y_test_pred.shape[1]):
            if Y_test_pred[i, j] > threshold_vals[i]:
                Y_test_pred_binary[i, j] = 1
            else:
                Y_test_pred_binary[i, j] = 0
   
    return Y_test_pred_binary, fit

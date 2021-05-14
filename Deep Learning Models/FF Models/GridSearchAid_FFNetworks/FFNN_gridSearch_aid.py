########################################################################################################################
# Filename: gridSearch_forFFNNs.py
#
# Purpose: Library to perform a hyperparameter grid search for feed forward artificial neural networks

# Author(s): Bobby (Robert) Lumpkin
#
# Library Dependencies: numpy, pandas, scikit-learn, skmultilearn, joblib, os, sys, threshold_learning
########################################################################################################################

import math
import tensorflow as tf
import tensorflow_addons as tfa
####################################

## Define a function that generates a list of layer sizes for a network
def SizeLayersPows2(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []
    
    first_layer_exp = math.log(first_layer_nodes, 2)
    last_layer_exp = math.log(last_layer_nodes, 2)
    exp_increment = (last_layer_exp - first_layer_exp) / (n_layers - 1)
    nodes = first_layer_nodes
    nodes_exp = first_layer_exp
    for i in range(1, n_layers):
        layers.append(nodes)
        nodes_exp = nodes_exp + exp_increment
        nodes = 2 ** round(nodes_exp)
    layers.append(last_layer_nodes)
    
    return layers

## Define a function which returns a network with a specified number of layers
def createModel(X_train, n_layers, first_layer_nodes, last_layer_nodes, activation_func, output_activation, loss_func, learning_rate = 0.01, Dropout_reg = True, drop_prob = 0.5):
    n_nodes = SizeLayersPows2(n_layers, first_layer_nodes, last_layer_nodes)
    layers_string = f"tf.keras.models.Sequential(["
    for layer in range(2, n_layers):
        layers_string = layers_string + f"tf.keras.layers.Dense({n_nodes[layer - 1]}, activation = \'" + activation_func + f"\'),"
    layers_string = layers_string + f"tf.keras.layers.Dense({last_layer_nodes}, activation = \'" + output_activation + "\')])"
    
    model = eval(layers_string)
    optim_func = tf.keras.optimizers.Adam(lr = learning_rate)
    model.compile(optimizer = optim_func, loss = loss_func, metrics = tfa.metrics.HammingLoss(mode = 'multilabel', threshold = 0.5)) 
    return model
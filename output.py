# coding: UTF-8
from unittest import expectedFailure
import numpy as np
import math


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(end_x):
    c = np.max(end_x)
    exp_x = np.exp(end_x - c)
    sum_exp_x = np.sum(end_x)
    output_x = exp_x /sum_exp_x
    return output_x

def mini_batch(x_train, t_train, batch_size):
    x_train_size = x_train.shape[0]
    use_data_index = np.random.choice(x_train_size, batch_size)
    x_train_used = x_train[use_data_index]
    t_train_used = t_train[use_data_index]
    print("--- index ---")
    print(use_data_index)
    print("--- xtrain used ---")
    print(x_train_used)
    print("--- ttrain used ---")
    print(t_train_used)
    return [x_train_used, t_train_used]

def OneLayerCal(x, W, b):
    a = np.dot(x, W) + b
    next_x = sigmoid(a)
    return next_x

# main exported fuction
def AllLayerCal(ini_x, network):
    network_length = len(network)-1
    network[0]['x'] = ini_x

    for i in range(network_length):
        network[i+1]['x'] = OneLayerCal(network[i]['x'], network[i]['W'], network[i]['b'])

    output = softmax(network[network_length]['x'])
    return output



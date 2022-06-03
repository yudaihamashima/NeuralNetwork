# coding: UTF-8
import numpy as np
import math


# error
def square_error(res, ans):
    return 0.5 * np.sum((res - ans)**2)


def closs_entropy_error(res, ans):
    delta = 1e-7
    return -np.sum(ans * np.log(res + delta))


def mean_error(result, answer, error_type):
    sum_error = 0
    for i in range(len(result)):
        sum_error += error_type(result[i], answer[i])
    return sum_error / len(result)





"""""
def one_layer_gradient(f, W):
    h = 1e-4
    grad = np.zeros_like(W)

    for i in range(W.size):
        x = W[i]
        grad[i] = (f(x+h) - f(x-h)) / (2*h)

    return grad


def netx_step_network(f, network):

    learning_rate = 0.01

    for i in range(len(network)):
        grad_W = one_layer_gradient(f, network[i]['W'])
        grad_b = one_layer_gradient(f, network[i]['b'])
        network[i]['W'] = network[i]['W'] - grad_W * learning_rate
        network[i]['b'] = network[i]['b'] - grad_b * learning_rate

    return network

"""
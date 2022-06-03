#coding: UTF-8
import numpy as np
import math

#順方向計算
import output

#学習
import error_evaluation



def enpty_samesize_network(network):
    x = {}
    layer_num = len(network)-1
    for i in range(layer_num):
        x[i] = {
            'W': np.zeros((len(network[i]['W']),len(network[i]['W'][0]))),
            'b': np.zeros(len(network[i]['b']))
        }
    x[len(network)] = {}
    return x


def network_error(network, x_train_lerning, t_train_lerning, error_function):
    result =  output.AllLayerCal(x_train_lerning, network)
    error = error_evaluation.mean_error(result, t_train_lerning, error_function)
    return error    


def network_gradient(network, x_train_lerning, t_train_lerning, error_function, grad_step):

    h = grad_step
    grad = enpty_samesize_network(network)
    net = network

    for i in range(len(network)-1):

        print("--- ","Layer ",i+1,'/',len(network)-1," gradient ---")
        W_i = net[i]['W']
        b_i = net[i]['b']

        print("W[",i,"] : ",W_i.size,"(",len(W_i),",",len(W_i[0]),")")
        for j in range(len(W_i)):
            for k in range(len(W_i[j])):
                
                print('W:',(j*len(W_i[j]))+k+1,' /',len(W_i)*len(W_i[j]), '\r', end='')

                W_ij2 = W_i[j][k] + h
                W_ij1 = W_i[j][k] - h

                net[i]['W'][j][k] = W_ij2
                error_2 = network_error(net, x_train_lerning, t_train_lerning, error_function)

                net[i]['W'][j][k] = W_ij1
                error_1 = network_error(net, x_train_lerning, t_train_lerning, error_function)

                net[i]['W'][j][k] = W_i[j][k]
                grad[i]['W'][j][k] = (error_2 - error_1) / (2*h)

        print("b[",i,"] : ",b_i.size)
        for j in range(len(b_i)):

            b_ij2 = b_i[j] + h
            b_ij1 = b_i[j] - h

            net[i]['b'][j] = b_ij2
            error_2 = network_error(net, x_train_lerning, t_train_lerning, error_function)

            net[i]['b'][j] = b_ij1
            error_1 = network_error(net, x_train_lerning, t_train_lerning, error_function)

            net[i]['b'][j] = b_i[j]
            grad[i]['b'][j] = (error_2 - error_1) / (2*h)

    return grad


def update_network(network, grad, learning_step):

    learning_rate = learning_step
    new_network = enpty_samesize_network(network)

    for i in range(len(network)-1):
        new_network[i]['W'] = network[i]['W'] - grad[i]['W'] * learning_rate
        new_network[i]['b'] = network[i]['b'] - grad[i]['b'] * learning_rate

    return new_network
        

   
#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import math

#ネットワークの初期値
import ini_network
network = ini_network.network
print(network[1]['W'])

#順方向計算
import output

#学習
import error_evaluation

#ネットワークのアップデート
import network_update

#訓練データ・テストデータ
import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=True)



################################ Learning ##################################

learning_time_string = input("Input Learning Times : ")
batch_size_string = input("Input Batch Size : ")

learning_time = int(learning_time_string)
batch_size = int(batch_size_string)

# changeable parameters
error_function = error_evaluation.closs_entropy_error
grad_step = 1e-4
learning_step = 0.1

errors = []
sequence = []

for i in range(learning_time):

    print("==========================")
    print("Learning Step :",i+1,"/",learning_time)
    print("\n")
    x_train_lerning = output.mini_batch(x_train, t_train, batch_size)[0]
    t_train_lerning = output.mini_batch(x_train, t_train, batch_size)[1]

    result =  output.AllLayerCal(x_train_lerning, network)
    error = error_evaluation.mean_error(result, t_train_lerning, error_function)
    
    errors.append(error)
    sequence.append(i)

    plt.xlabel('Learning times')
    plt.ylabel('Error')
    plt.plot(sequence, errors, color='blue')
    plt.pause(1)
    
    print("---- Expected Nummber ----")
    print(result)
    print(np.argmax(result, 1))
    print("---- Accurate Nummber ----")
    print(np.argmax(t_train_lerning, 1))
    print("---------- Error ---------")
    print(error)
    print("\n")

    grad = network_update.network_gradient(network, x_train_lerning, t_train_lerning, error_function, grad_step)
    next_network = network_update.update_network(network, grad, learning_step)
    net_work = next_network
    print("---- network updated -----")
    print("==========================")
    print("\n")
    print("\n")


print("========= Result =========")
x_train_lerning = output.mini_batch(x_train, t_train, batch_size)[0]
t_train_lerning = output.mini_batch(x_train, t_train, batch_size)[1]
result =  output.AllLayerCal(x_train_lerning, network)
error = error_evaluation.mean_error(result, t_train_lerning, error_function)

errors.append(error)
sequence.append(learning_time)

print("---- Expected Nummber ----")
print()
print(np.argmax(result, 1))
print("---- Accurate Nummber ----")
print(np.argmax(t_train_lerning, 1))
print("---------- Error ---------")
print(error)
print("\n")

plt.xlabel('Learning times')
plt.ylabel('Error')
plt.plot(sequence, errors, color='red')
plt.pause(1000)




################################ Evaluation ##################################








""""
########### 順方向 ############

#data random choice
batch_size = 5
x_train_lerning = output.mini_batch(x_train, t_train, batch_size)[0]
t_train_lerning = output.mini_batch(x_train, t_train, batch_size)[1]

#caluculation
result =  output.AllLayerCal(x_train_lerning, network)

#display
print("---- result ----")
print(result)

print("---- expectation ----")
print(np.argmax(result, 1))

print("---- true letter ---")
print(np.argmax(t_train_lerning, 1))

print(len(x_train_lerning))



########### 誤差 ############

error = error_evaluation.mean_error(result, t_train_lerning, error_evaluation.closs_entropy_error)
print(error)


print("====================== \n")

"""

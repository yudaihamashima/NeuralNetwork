# coding: UTF-8
#ネットワークの初期値
import numpy as np

#input          :784(28*28)
#hiddn layer 1  :50
#hiddn layer 2  :100
#output         :10

network = {}

#784*50
network[0] = {
    'W': np.random.rand(784, 50)*0.01,
    #'b': np.random.rand(50)*0.01
    'b': np.zeros(50)*0.01
}

#50*100
network[1] = {
    'W': np.random.rand(50, 100)*0.01,
    #'b': np.random.rand(100)*0.01
    'b': np.zeros(100)*0.01
}

#100*10
network[2] = {
    'W': np.random.rand(100, 10)*0.01,
    #'b': np.random.rand(10)*0.01
    'b': np.zeros(10)*0.01
}

# only 'x' : fot output
network[3] = {}



"""
network = {}
network[0] = {'W': np.array([[1,2,3],
                             [4,5,6]]),

              'b': np.array([1,2,3])
             }
network[1] = {'W': np.array([[2,3,4],
                             [5,6,7],
                             [8,9,10]]),

              'b': np.array([2,3,4])
             }
network[2] = {'W': np.array([[2,3],
                             [5,6],
                             [8,9]]),

              'b': np.array([2,3])
             }
             
# only 'x' : fot output
network[3] = {}

"""

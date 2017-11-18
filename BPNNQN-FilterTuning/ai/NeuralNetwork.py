# -*- coding: utf-8 -*-
"""
NeuralNet class creates a neural network.
@author: Jingfeng Yang 2016-03-10 create
@author: Jingfeng Yang 2016-03-25 modify

"""
#from convnet import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import  Dense, Dropout, Activation, Flatten
#from keras.utils import np_utils
from keras.optimizers import  RMSprop, SGD


class Nnet:
    nnet = []

    def __init__(self):
    	"""
    	Networks
    	@ input layer: 20
    	@ hidden layer: 50, activation = sigmoid
    	@ output layer: 4, activation = linear
    	@ optmizer: rmsprop
    	@ loss: mean squared error
    	"""
        self.nnet = Sequential()
        #sgd = SGD(lr = 0.01,decay = 1e-6, momentum = 0.9, nesterov = True)
        rmsprop = RMSprop(lr = 0.002,rho = 0.9, epsilon = 1e-6)
        
        self.nnet.add(Dense(output_dim = 50, input_dim = 20))
        self.nnet.add(Activation('sigmoid'))
        self.nnet.add(Dense(output_dim = 4, input_dim = 50))
        self.nnet.add(Activation('linear'))
        self.nnet.compile(loss = 'mean_squared_error', optimizer = rmsprop)

    def net_train(self, minibatch, myfilter, discount_factor):
    	"""
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: list of arrays: prestates, actions, rewards, poststates
        """
        
        prestates, actions, rewards, poststates = minibatch

        # dimensionality reduction
        prepcastates = myfilter.dimreduction_pca(prestates)
        postpcastates = myfilter.dimreduction_pca(poststates)

        # predict Q-values for prestates, so we can keep Q-values for other actions unchanged
        qvalues = self.nnet.predict(prepcastates)

        # predict Q-values for poststates
        post_qvalues = self.nnet.predict(postpcastates)

        # take maximum Q-value of all actions
        max_qvalues = np.max(post_qvalues, axis = 1)

        # update the Q-values for the actions we actually performed
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + discount_factor * max_qvalues[i]

        # we have to transpose prediction result, as train expects input in opposite order
        self.nnet.fit(prepcastates, qvalues,nb_epoch=500, batch_size=1000, validation_split = 0.1, verbose = 0)

        return self.nnet
# ---- The End -----#
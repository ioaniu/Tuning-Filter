# -*- coding: utf-8 -*-
"""
Train a model for simulating tuning filter.
@author: Jingfeng Yang 2016-03-10 
"""

"""
save as: simulate_architechture.json
save as: simulate_architechture.yaml
save as: simulate_weigths.h5
save as: pca

when call:

#import pickle
#from keras.models import  model_from_json, model_from_yaml
#model =  model_from_json(open('simulate_architechture.json').read())
#model =  model_from_yaml(open('simulate_architechture.yaml').read())
#model.load_weights('simulate_weigths.h5')
# f = open('pca','r')
# eigenmatrix = pickle.load(f)
# f.close()




"""

import scipy.io
import pickle

import numpy as np
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn

from keras.models import Sequential
from keras.layers.core import  Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import  RMSprop, SGD

import matplotlib.pyplot as plt
import random
import drawnow

from sklearn.decomposition import PCA

class Simulate:
  
    

    def __init__(self):
        self.nnet = Sequential()
        #sgd = SGD(lr = 0.1,decay = 1e-6, momentum = 0.9, nesterov = True)
        rmsprop = RMSprop(lr = 0.1,rho = 0.9, epsilon = 1e-6)
        
        self.nnet.add(Dense(output_dim = 20, input_dim = 2))
        self.nnet.add(Activation('sigmoid'))
        self.nnet.add(Dense(output_dim = 100, input_dim = 20))
        self.nnet.add(Activation('sigmoid'))
        self.nnet.add(Dense(output_dim = 20, input_dim = 100))
        self.nnet.add(Activation('linear'))
        self.nnet.compile(loss = 'mean_squared_error', optimizer = rmsprop)

    def cov(self, data):
        """
        covariance matrix
        """
        N = data.shape[1]
        C = empty((N,N))
        
        for j in range(N):
            C[j,j] = mean(data[:,j] * data[:,j])
            for k in range(N):
                C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
        return C
    
#    def PCA(self, data, pca_count = None):
#        
#        data -= mean(data, 0)
#        data /= std(data, 0)
#        C = self.cov(data)
#        E, V =eigh(C)
#        key = argsort(E)[::-1][:pca_count]
#        E, V = E[key], V[:, key]
#        U = dot(V.T, data.T).T
#        return U, E, V
    def train_data(self,inputdata, outputdata):
        
        loss = self.nnet.fit(inputdata,outputdata, nb_epoch = 5000, batch_size = 5000)
        
        return nnet
#    def train_data(self,sparams, screws):
#        inputdata = screws
#        datatemp = sparams
#        pca = PCA(n_components = 20)
#        
#        pcadata = pca.fit(datatemp)
##        U, E, V = self.PCA(datatemp)        
##        outputdata = np.dot(sparams, V[:,0:20])
#        outputdata = np.dot(sparams, pcadata.components_.transpose())
#        
#        loss = self.nnet.fit(inputdata, outputdata, nb_epoch = 5000, batch_size = 5000)
#        
#        
#        self.nnet.save_weights('simulate_weigths.h5')
#        open('simulate_architechture.json', 'w').write(self.nnet.to_json())
#        open('simulate_architechture.yaml', 'w').write(self.nnet.to_yaml())
#        f = open('pca','w')
#        pickle.dump(pcadata.components_.transpose(), f)
#        f.close()
#        return self.nnet, pcadata.components_.transpose()
    
if __name__ == '__main__':
    
    data = scipy.io.loadmat('simulatedata.mat')
     
    pcadata = data.get('pcadata')
    pcaeigen = data.get('pcaeigen')
    pcamean = data.get('pcamean')
    screws = data.get('screws')
    #data = scipy.io.loadmat('data1.mat')
    #sparams = data.get('s_params').transpose()
    #screws = data.get('screws')
    #sparams = np.transpose(data['s_params'])
    #screws = data['screws']    
    model = None
    
    run = Simulate()
    
#    nnet,eignpca = run.train_data(sparams, screws)
    run.nnet.fit(screws,pcadata,nb_epoch = 5000, batch_size = 5000)
#    nnet = run.train_data(screws,pcadata)
    
    run.nnet.save_weights('simulate_weigths.h5')
    open('simulate_architechture.json', 'w').write(run.nnet.to_json())
    open('simulate_architechture.yaml', 'w').write(run.nnet.to_yaml())
    f = open('pcaeigen','w')
    pickle.dump(pcaeigen.transpose(), f)
    f.close()
    f = open('pcamean','w')
    pickle.dump(pcaeigen.transpose(), f)
    f.close()
    
        
    

    
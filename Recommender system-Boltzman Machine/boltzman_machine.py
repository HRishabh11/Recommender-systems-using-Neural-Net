# Creating the architecture of the Neural Network


# importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable


# inputs
    # nb_v = # of visible nodes
    # nb_h = # of hidden nodes
# output
    #

class RBM():
    def __init__(self, nb_v, nb_h):
        self.W = torch.randn(nb_h,nb_v)       # initializing the weights
        self.bias_h = torch.randn(1,nb_h)     # initializing the bias for hidden nodes
        self.bias_v = torch.randn(1,nb_v)     # initializing the bias for visible nodes

    # sampling method for hidden neurons
    def sample_h(self,x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.bias_h.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)         # probability of h given v
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    # sampling method for visible nodes
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.bias_v.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)         # probability of v given h
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # method to update the weights by using contrastive divergence
    def update(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t() 
        self.bias_v += torch.sum((v0 - vk),0)
        self.bias_h += torch.sum((ph0 - phk),0)
        
    # method to train the RBM
    def train(self, nb_epoch,nb_users, batch_size, training_set, k):
        for epoch in range(1,nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id in range(0, nb_users - batch_size,batch_size):
                vk = training_set[id: id + batch_size]
                v0 = training_set[id: id + batch_size]
                ph0,_ = self.sample_h(v0)
                for k in range(k):
                    _,hk = self.sample_h(vk)
                    _,vk = self.sample_v(hk)
                    vk[v0 < 0] = v0[v0 < 0]
                phk,_ = self.sample_h(vk)
                self.update(v0, vk, ph0, phk)
                train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
                s += 1.
            print('epoch: ' + str(epoch) + ' loss: ' + str(int(train_loss)/s))
            
    def evaluate(self, training_set, test_set):
        test_loss = 0
        s = 0.
        for id in range(0, nb_users):
            v = training_set[id: id + 1]
            vt = test_set[id: id + 1]
            if len(vt[vt >= 0]) > 0:
                _,h = self.sample_h(v)
                _,v = self.sample_v(h)
            self.update(v0, vk, ph0, phk)
            test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
            s += 1.
        print('test loss: ' + str(int(test_loss)/s))
            
            
            
            
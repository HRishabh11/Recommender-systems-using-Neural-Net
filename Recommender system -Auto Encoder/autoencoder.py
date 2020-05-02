#Auto Encoder

#importing required libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import variable


#an object to preprocess the given data into required format
class Preprocessing:
    def __init__(self):
        pass
     
    # method to covert the data
    # inputs
        # data = dataset to be converted into required format
        #        1st column should be user ids
        #        2nd column should be movie ids
        #        3rd column should be ratings
        # n    = number of users
        # k    = number of movies
    # output:
        # new_data = coverted dataset
        #            list of lists
        #            each list contains the ratings given by each users
        #            if user hasn't rated the movie a 0 rating is assigned
    def convert(self,data,n,k):
        new_data = []
        for id in range(1,n + 1):
            id_movies = data[:,1][data[:,0] == id]
            id_ratings = data[:,2][data[:,0] == id]
            ratings = np.zeros(k)
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        return new_data
    
    # method that uses the convert method to actually tranform the data
    # inputs:
        # train = training dataset
        # test = test dataset
    # outputs:
        # newtrain = converted train dataset
        # newtest = converted test dataset
    def processed(self,train,test):
        nb_users = max(max(train[:,0]),max(test[:,0]))
        nb_movies = max(max(train[:,1]),max(test[:,1]))
        newtrain = self.convert(train,nb_users,nb_movies)
        newtest = self.convert(test,nb_users,nb_movies)
        newtrain = torch.FloatTensor(newtrain)
        newtest = torch.FloatTensor(newtest)
        return newtrain, newtest
         
    
# object to create autoencoder architecture
class AutoEncoder(nn.Module):
    def __init__(self, nb_movies, nodes1, nodes2):
        
        # inheritance from Module object of torch.nn
        super(AutoEncoder,self).__init__()
        
        # Network Layers
        self.fc1 = nn.Linear(nb_movies, nodes1)
        self.fc2 = nn.Linear(nodes1,nodes2)
        self.fc3 = nn.Linear(nodes2,nodes1)
        self.fc4 = nn.Linear(nodes1, nb_movies)
        
        # Activation function
        self.activation = nn.Sigmoid()
        
    # method to train each row of the data
    # input:
        # x = rows of training data
    # output:
        # x = predictions for the given row
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
    # method to train whole data
    def train(self, training, test, learning_rate, wt_decay, nb_epoch):
        p = Preprocessing()
        training,_ = p.processed(training, test)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr = learning_rate,
                                  weight_decay = wt_decay)
        for epoch in range(1,nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id in range(0,len(training)):
                input = variable(training[id]).unsqueeze(0)
                target = input.clone()
                if torch.sum(target.data > 0) > 0:
                    output = self.forward(input)
                    target.required_grad = False
                    output[target == 0] = 0
                    loss = criterion(output, target)
                    mean_corrector = training.size()[1]/float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.data*mean_corrector)
                    s += 1.
                    optimizer.step()
            print('Epoch: '+str(epoch) + ' Loss: ' + str(int(train_loss)/s))
            
    # method to test the model
    def model_test(self,training, test):
        criterion = nn.MSELoss()
        p = Preprocessing()
        training,test = p.processed(training, test)
        test_loss = 0
        s = 0.
        for id in range(0,len(training)):
            input = variable(training[id]).unsqueeze(0)
            target = variable(test[id]).unsqueeze(0)
            if torch.sum(target.data > 0) > 0:
                output = self.forward(input)
                target.require_grad = False
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = training.size()[1]/float(torch.sum(target.data > 0) + 1e-10)
                test_loss += np.sqrt(loss.data*mean_corrector)
                s += 1.
        print('Loss: ' + str(int(test_loss)/s))
                    
    # method to make prediction
    def predict(self,training,test):
        p = Preprocessing()
        data,_ = p.processed(training,test)
        output = []
        for id in range(0,len(data)):
            input = variable(data[id]).unsqueeze(0)
            pred = self.forward(input)
            output.append(pred.detach().numpy())
        return output
        
        
        
        
        
        
        
        
        
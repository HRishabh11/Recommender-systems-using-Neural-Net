# importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable

# importing the dataset
movies = pd.read_csv("ml-1m/movies.dat", sep = '::', header = None,
                     engine = 'python', encoding = 'latin-1')

users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None,
                    engine = 'python', encoding = 'latin-1')

ratings = pd.read_csv("ml-1m/ratings.dat",sep = '::', header = None,
                      engine = 'python', encoding = 'latin-1')


# creating the training and test set
training_set = pd.read_csv("ml-100k/u1.base", delimiter = '\t').values
test_set = pd.read_csv("ml-100k/u1.test", delimiter = '\t').values

# getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id in range(1,nb_users + 1):
        id_movies = data[:,1][data[:,0] == id]    # getting all the movies reviewed
                                                # by the user of having id
        id_ratings = data[:,2][data[:,0] == id]   # getting all the ratings rated by
                                                # the users
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings     # creating the list of ratings by rated 
                                                # by user id 
        new_data.append(list(ratings))
    return new_data

#converting the training set and the test set
training_set = convert(training_set)
test_set = convert(test_set)


# converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(training_set)

# converting the ratings into binary ratings 1 (liked) or 0 (Not liked)
# since the unratred movie had a value of zero, we must change that

# converting the previous 0 (not rated) value to -1
training_set[training_set == 0] = -1   
test_set[test_set == 0] = -1

# converting the rated movies to have a value 0 (Not liked) if rating is less than 2 
training_set[training_set <= 2] = 0
test_set[test_set <= 2] = 0

# converting the rated movies to have a value 1 (liked) if rating is greate than 2 
training_set[training_set > 2] = 1
test_set[test_set > 2] = 1


# no. of visible nodes
nb_v = len(training_set[0])

# no. of hidden nodes
nb_h = 100

# batch size 
batch_size = 100

# creating the RBM architecture
import boltzman_machine
rbm = RBM(nb_v,nb_h)   

# Training the RBM
rbm.train(10,nb_users,100,training_set,10)

#evaluating the model
rbm.evaluate(training_set,test_set)

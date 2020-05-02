# Autoencoder data preprocessing

# importing libraries
import pandas as pd 
import torch
import torch.nn.parallel

#importing the data
movies = pd.read_csv("ml-1m/movies.dat", sep = '::',header = None,
                     encoding = 'latin-1')
users = pd.read_csv("ml-1m/users.dat",sep = "::",header = None,
                    encoding = 'latin-1' )
ratings = pd.read_csv("ml-1m/ratings.dat",sep = "::", header = None,
                      encoding = 'latin-1')


#Feature engineering
training_set = pd.read_csv("ml-100k/u1.base", delimiter = '\t').values
test_set = pd.read_csv("ml-100k/u1.test",delimiter = '\t').values


#creating the architecture
from autoencoder import AutoEncoder
nb_movies = max(max(training_set[:,1]),max(test_set[:,1]))
sae = AutoEncoder(nb_movies,20,10)
sae.train(training_set,test_set,0.01,0.5,200)

#Training the model
sae.model_test(training_set,test_set)

#making predictions
pred = sae.predict(training_set, test_set)


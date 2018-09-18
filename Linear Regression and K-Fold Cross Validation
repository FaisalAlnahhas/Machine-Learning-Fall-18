
# coding: utf-8

# In[20]:

############################################
#Faisal Alnahhas
#k-fold cross validation linear regression
#UTA - Fall'18
#CSE 6363 - Machine Learning
#Assignment 1
############################################


# In[1]:

#Array processing
import numpy as np
#Data analysis, wrangling and common exploratory operations
import pandas as pd
import random
from sklearn.model_selection import KFold


# In[2]:
print("Welcome to the linear regression training model for the Iris Data\n")
print("In this program you can choose a 5 or 10 fold cross validation to obtain accuracy for predicted output.\n")
print("The data can be obtained from: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
print("Please follow the prompt to see results.\n")
print("Note: to test this file on other .csv data you need to have the new file in the same directory ")
print("as this .py file, in addition you would need to update the name of the file in this program.")

user_input = input("How many folds for nfold cross validation, please type in 5, or 10?\n")
while user_input not in ('5', '10'):
    print("This program only supports 5 and 10 fold cross validations\n")
    user_input = input("How many folds for nfold cross validation, please type in 5, or 10?\n")
folds = int(user_input)

df = pd.read_csv('iris.data.csv', header=None)
df = df.values
Z = df[:, :4]
Z = Z.astype(np.float64)
Y = df[:, 4:5]
for i in range(50):
    Y[i] = 1
for j in range(50,100):
    Y[j] = 2
for j in range(100,150):
    Y[j] = 3
Y = Y.astype(np.float64)
print(df.shape, Z.shape, Z.T.shape, Y.shape)
#print(Y)


# In[3]:

SSE_values = []
beta_values = []


# In[21]:


#print(type(folds))


# In[22]:

if folds == 5:
    set_size = 30
else:
    set_size = 15


# In[32]:

#Generate a uniform random sample from the list of inputs
def cross_validation():
    index_set = []
    test_set = []
    test_set_indeces = []
    test_set_labels = []
    train_set = []
    train_set_indeces = []
    train_set_labels = []
    error = []
    for i in range(len(Z)):
        index_set.append(i)
    #print(index_set)
    #print("test set indeces\n")
    test_set_indeces = random.sample(index_set, set_size)
    #print("test set indeces: " + str(test_set_indeces))
    print("length of test set\n")
    print(len(test_set_indeces))


    #print("their corresponding index values from Z\n")
    for j in test_set_indeces:
        test_set.append(Z[j])
        test_set_labels.append(Y[j])
    test_set = np.asmatrix(test_set)
    print(test_set.shape)
    #print("test set: "+ str(test_set))


    #Train set
    print("train set indeces length\n")
    for k in index_set:
        if k not in test_set_indeces:
            train_set_indeces.append(k)
    print(len(train_set_indeces))

    for j in train_set_indeces:
        train_set.append(Z[j])
        train_set_labels.append(Y[j])
    train_set = np.asmatrix(train_set)
    print(train_set.shape)

    ZTZInv = np.linalg.inv(np.dot(train_set.T, train_set))
    ZTY = np.dot(train_set.T, train_set_labels)
    beta = np.dot(ZTZInv, ZTY)
    beta_values.append(beta)
    ##np.dot(beta, test_set_y_values) compute y_hat
    y_hat = np.dot(test_set, beta)
    
    #rounding output:
    for i in range(len(y_hat)):
        if y_hat[i] <= 1.499:
            y_hat[i] = 1
        elif 1.5<=y_hat[i]<=2.499:
            y_hat[i] = 2
        else:
            y_hat[i] = 3
    print("Shape of beta, shape of y_hat")    
    print(beta.shape, y_hat.shape)
    #print("y_hat:\n"+str(y_hat))
    
    ##computer error
    for i in range(len(y_hat)):
        error.append((y_hat[i] - test_set_labels[i])**2)
    SSE = (1/len(y_hat)*np.sum(error))*100
    SSE_values.append(SSE)
    print("SSE = " +str(SSE))
cross_validation()


# In[33]:

##Source https://stackoverflow.com/questions/9047985/how-do-i-call-a-function-twice-or-more-times-consecutively-in-python
def repeat_function(times, f):
    for i in range(times): f()
repeat_function(folds, cross_validation)


# In[35]:

#Average SSE
SSE_Av = np.average(SSE_values)
SSE_best = np.min(SSE_values)
#print(SSE_Av, SSE_best)
print("\n\n\n")
for i in range(len(SSE_values)):
    if SSE_values[i] == SSE_best:
        beta_best = beta_values[i]
print("From the " +str(folds)+"-fold cross validation we conclude the following: \n")
print("SSE minimum value = " + str(SSE_best) + "%\n")
print("SSE average = " + str(SSE_Av) + "%\n")
print("Best beta matrix associated with SSE minimum value = \n" + str(beta_best) + "\n")
print("Thank you for using this program")


# In[ ]:




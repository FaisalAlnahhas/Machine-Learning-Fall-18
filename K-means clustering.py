############################################
#Faisal Alnahhas
#k-means clustering
#UTA - Fall'18
#CSE 6363 - Machine Learning
#Assignment 3
############################################

# coding: utf-8

# In[23]:

import numpy as np
from sklearn import datasets
import pandas as pd
import random


# In[2]:

data = datasets.load_iris()
labels = data.target
print(type(data))


# In[24]:

print("Welcome to the k-means clustering for the Iris Data\n")
print("In this program you can choose any number between 1 and 150 (size of data set) to obtain accuracy for predicted output.\n")
print("The data can be obtained from: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
print("Or it can be imported from sklearn.datasets\n")
print("Please follow the prompt to see results.\n")


# In[3]:

df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= data['feature_names'] + ['target'])
df = df.values
type(data)
print(type(df))


# In[25]:

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


# In[6]:

num_clusters = int(input("How many clusters would you like? Please type a number between 1 and 10 \n"))
if num_clusters<1 or num_clusters>10 :
    print("You can't have less than 1 cluster or greater 10: \n")
    num_clusters = int(input("How many clusters would you like? Please type a number between 1 and 10 \n"))


# In[7]:

def random_generator(size, start, finish):
    random_indeces = []
    for i in range(size):
        random_indeces.append(random.randint(start, finish))
    return random_indeces
# random_generator(10, -10, 13)


# In[8]:

def euclidean_distance(centers, data):
    norms = []
    for i in range(len(centers)):
        norms.append(np.linalg.norm(data-centers[i]))
    return norms
# euclidean_distance(centroids, Z)



def make_centroids(num_clusters):
    centeroid_indeces = random_generator(num_clusters, 0, len(Z))
#     print(centeroid_indeces)
#     print(len(centeroid_indeces))
    centroids = []
    for i in centeroid_indeces:
        centroids.append(Z[i])
    return centroids
centroids = make_centroids(num_clusters)


# In[10]:

def calc_distances(centers):
    distances = {}
    distance_list = []
    min_index = []
    for i in range(len(Z)):
        distance = euclidean_distance(centroids, Z[i])
        distance_list.append(min(distance))
        min_index.append(np.argmin(distance))
#     print((min_index))
#     print(distance_list)
    return distances, distance_list, min_index
# distances, distance_list, min_index = calc_distances(centroids)


# In[11]:

def make_clusters(distance_list, min_index):   
    clusters = {}
    #unique elements in indeces to make dictionary
    #if index[i] in dictionary, append element
    unique = np.unique(min_index)
    for i in unique:
        clusters[unique[i]] = []
        
    for k, v in clusters.items(): 
        for i,j in zip(range(len(distance_list)), range(len(min_index))):
            if min_index[j] == k:
                v.append(distance_list[i])
    return clusters
# clusters = make_clusters(distance_list, min_index)


# In[12]:

def cluster_size(cluster_dict):
    clusters_size_list = []
    for k,v in cluster_dict.items():
        s = len(v)
        clusters_size_list.append(s)
    return clusters_size_list
# length_clusters = cluster_size(clusters)
        


# In[13]:

def calc_accuracy(t, a):
    error = 0
    t.sort()
#     print(t)
    for i in range(len(t)):
        if t[i] != a[i]:
            error += 1
    error_percentage = (error/len(a))*100
    print("accuracy = ", error_percentage, "%")
    return error_percentage
# calc_accuracy(min_index, Y)



def update_centroids(clusters_dictionary):
    for k,v in clusters_dictionary.items():
        new_cluster = np.average(v)
        clusters_dictionary[k] = new_cluster
    return clusters_dictionary
# new_centroids = update_centroids(clusters)




for i in range(num_clusters):
    distances, distance_list, min_index = calc_distances(centroids)
    clusters = make_clusters(distance_list, min_index)
    length_clusters = cluster_size(clusters)
    new_centroids = centroids
    new_centroids = update_centroids(clusters)
calc_accuracy(min_index,Y)


# In[ ]:




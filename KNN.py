__authors__ = ['1674585','1673987','1674822']
__group__ = 'DL.17'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # self.train_data = np.random.randint(8, size=[10, 4800])
        P = train_data.shape[0]
        self.train_data = train_data.reshape(P, -1)
        self.train_data = self.train_data.astype(float)
        return self.train_data

    def get_k_neighbours(self, test_data, k, type_distance="euclidean"):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = test_data.astype(float)
        test_data = test_data.reshape(test_data.shape[0], -1)
        distancia = cdist(test_data, self.train_data, type_distance)

        posicio = np.argsort(distancia, axis=1)[:, :k]
        self.neighbors = self.labels[posicio]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        vots = []

        for i in self.neighbors:
            no_repetits, repetits = np.unique(i, return_counts=True)

            if len(i) != len(no_repetits):

                for etiqueta in i:

                    if list(i).count(etiqueta) == np.max(repetits):
                        vots.append(etiqueta)
                        break
            else:
                vots.append(i[0])

        return np.array(vots)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
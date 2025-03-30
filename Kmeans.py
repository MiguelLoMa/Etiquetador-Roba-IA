__authors__ = ['1674585','1673987','1674822']
__group__ = 'DL.17'

import numpy as np
import utils
import scipy


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        X = np.array(X, dtype=float)  # a)

        if X.ndim == 3 and X.shape[2] == 3:  # b) Si son 3 dimensions i si F x C x 3
            X = X.reshape(-1, 3)  # N x 3
        elif X.ndim > 2:  # c) Converteix les dimensions a 2 --> N x D.
            X = X.reshape(X.shape[0], -1)

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.centroids = np.zeros((self.K, self.X.shape[1]))
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))

        if self.options['km_init'].lower() == 'first':
            # Asignar la primera fila de self.X a la primera fila de centroids
            self.centroids[0] = self.X[0]

            # Inicializar contador
            contK = 0

            # Iterar sobre las filas de self.X
            for fila in self.X[1:]:
                # Verificar si la fila actual está en centroids
                esta_en_centroids = np.any(np.all(fila == self.centroids[:contK + 1], axis=1))

                if not esta_en_centroids:
                    # Si la fila no está en centroids, aumentar el contador contK y asignar la fila a centroids
                    contK += 1
                    self.centroids[contK] = fila

                # Verificar si se han alcanzado K filas únicas en centroids
                if contK == self.K - 1:
                    break

        elif self.options['random'].lower() == 'random':
            indices_aleatorios = np.random.choice(self.X.shape[0], size=self.K, replace=False)
            self.centroids = self.X[indices_aleatorios]
        elif self.options['last'].lower() == 'last':
            # Asignar la última fila de self.X a la primera fila de centroids
            self.centroids[0] = self.X[-1]

            # Inicializar contador
            contK = 0

            # Iterar sobre las filas de self.X en orden inverso
            for fila in self.X[-2::-1]:
                # Verificar si la fila actual está en centroids
                esta_en_centroids = np.any(np.all(fila == self.centroids[:contK + 1], axis=1))

                if not esta_en_centroids:
                    # Si la fila no está en centroids, aumentar el contador contK y asignar la fila a centroids
                    contK += 1
                    self.centroids[contK] = fila

                # Verificar si se han alcanzado K filas únicas en centroids
                if contK == self.K - 1:
                    break

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distancia = distance(self.X, self.centroids)
        self.labels = np.argmin(distancia, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.copy(self.centroids)
        for fila in range(self.centroids.shape[0]):
            # de cada fila calculem la mitja de nomes els punts que siguin de la mateixa clase
            self.centroids[fila] = np.mean(self.X[self.labels == fila], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if np.array_equal(self.centroids, self.old_centroids):
            return True
        else:
            return False

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()

        while not self.converges():

            #trobar el centroide mes proper a cada punt de la image
            self.get_labels()
            #calculem els nous centroides
            self.get_centroids()
            #finalment aumentem el nombre d'iteracions
            self.num_iter = self.num_iter + 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distancia = 0
        tamanyx = len(self.X)
        for i in range(self.K):

            punts = self.X[self.labels == i] #obtenim les etiquetes mes proximes als punts de la imatge x
            if len(punts) > 0:
                Centroide = self.centroids[i]

                #fem distancia euclidiana amb np
                distancies = np.linalg.norm(punts - Centroide, axis=1)
                distancia = distancia + np.sum(distancies ** 2) #fem el cuadrat de la distancia que hem obtingut
        distancia = distancia / tamanyx
        return distancia

    def interClassDistance(self):
        distancia = 0

        for i in range(len(self.centroids)):
            nuvol = self.X[self.labels == i]
            if len(nuvol) != 0:
                centroides = np.delete(self.centroids, i, axis=0)
                for centroide in centroides:
                    distancia += np.sum((nuvol - centroide) ** 2)
        return distancia / len(self.X)

    def fisherCoeficient(self):
        return self.withinClassDistance()/self.interClassDistance()

    def find_bestK(self, max_K, heuristiques = "WCD"):
        llista = []
        llistadistancies = []
        optim = 0

        for K in range(2, max_K + 1):
            self.K = K
            self.fit()
            if heuristiques == "WCD":
                llistadistancies.append(self.withinClassDistance())
            elif heuristiques == "ICD":
                llistadistancies.append(self.interClassDistance())
            elif heuristiques == "FC":
                llistadistancies.append(self.fisherCoeficient())

            if len(llistadistancies) > 1:
                llista.append(100 * (llistadistancies[-1]) / llistadistancies[-2])

                if len(llista) > 1:

                    if (100 - llista[-1]) < 20:
                        self.K = K - 1
                        optim = 1
                        break
        #en cas de que loptim sigui o retornem el maxk que es passa per defecte
        if optim == 0:
            self.K = max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    #obtenim la disctancia entre cada punt x y  le centroide C
    differences = X[:, np.newaxis, :] - C

    #calcula el cuadrat de la diferencia
    squared_diffs = differences ** 2

    #sumem el cuadrat de les diferencia en tot el llarg del eix per obtenir les distancies euclidianes
    dist = np.sqrt(np.sum(squared_diffs, axis=2))

    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    #copidm els centroides
    centroids = np.array(centroids)
    #obtenim les probabilitats de cada centroide amb utils get colot prob
    probabilitat_colors = utils.get_color_prob(centroids)
    maximaprobabilitat = np.argmax(probabilitat_colors, axis=1)
    #creem un vector de etiquetes
    etquietes = [utils.colors[i] for i in maximaprobabilitat]

    return (etquietes)
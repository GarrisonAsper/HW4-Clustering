import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        #error handling
        if not (k, int) or k < 0:
            raise ValueError('Invalid k, must be positive integer')
        if not(tol, (float, int)) or tol < 0:
            raise ValueError('Invalid tol, must be positive float or integer')
        if not (max_iter, int) or max_iter < 0:
            raise ValueError('Invalid max_iter, must be positive integer')
        
        #attributes to store inputs
        self.k = k 
        self.tol = tol
        self.max_iter = max_iter

        #further attributes to store states
        self.centroid = None
        self.error = None
        self.fitted = False #to determine if Kmean has been fit or not


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not (mat, np.ndarray) or len(mat.shape) != 2:
            raise ValueError('mat must be 2D numpy array')

        k = self.k
        tol = self.tol
        max_iter = self.max_iter

        #initializing k random centroids 
        centroids = np.empty((0,2))
        for _ in range(k):
            centroid = np.array([[np.random.uniform(mat.min(), mat.max()), np.random.uniform(mat.min(), mat.max())]])
            centroids = np.append(centroids, centroid, axis = 0)

        iter_count = 0
        centroid_delta = float("inf")

        #begin loop until max iterations or centroid_delta approaches tolerance
        while iter_count < max_iter and centroid_delta > tol:
            #initializing and resetting index_dict to track centroid assignment
            index_dict = {i: [] for i in range(k)}
            #calculating distances from matrix to centroid points
            distances = cdist(mat, centroids, metric = 'euclidean')
            #picks index of minimum distance
            bins = np.argmin(distances, axis = 1)
        
            #store index of each bin for assignment
            for i, value in enumerate(bins):
                index_dict[value].append(i)
            
            #initializing new centroids based off bin assignments
            new_centroids = np.empty_like(centroids)
            for key, indices in index_dict.items():
                if len(indices) > 0:
                    new_centroids[key] = np.mean(mat[indices], axis = 0)
                #randomly initiate centroid if no points are assigned to it
                else:
                    new_centroids[key] = np.array([np.random.uniform(mat.min(), mat.max()), np.random.uniform(mat.min(), mat.max())])
            
            #calculating difference between centroids and new centroids
            centroid_delta = np.linalg.norm(centroids - new_centroids)
            iter_count += 1
            #updating centroids
            centroids = new_centroids.copy()
        
        self.centroids = centroids #stores fitted centroids in callable attribute
        self.fitted = True #marks model as fitted
        self.error = np.mean(np.min(cdist(mat, centroids, metric = 'euclidean'), axis=1) ** 2) #calculates squared mean error


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.fitted == False:
            raise ValueError('Model not fitted: call fit()')
        if not (mat, np.ndarray) or len(mat.shape) != 2:
            raise ValueError('mat must be 2D numpy array')
        
        distances = cdist(mat, self.centroids, metric = 'euclidean') #calculates distance between points and centroids

        return np.argmin(distances, axis = 1) #takes minimum distance as assignment

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.fitted == False:
            raise ValueError('Model not fitted: call fit()')

        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.fitted == False:
            raise ValueError('Model not fitted: call fit()')
        
        return self.centroids
import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        #error handling
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("X must be a 2D NumPy array.")
        if not isinstance(y, np.ndarray) or len(y.shape) != 1:
            raise ValueError("y must be a 1D NumPy array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        

        unique_clusters = np.unique(y)  #find unique cluster labels
        n_samples = X.shape[0]  #number of data points
        silhouette_scores = np.zeros(n_samples)  #initialize silhouette scores

        for i in range(n_samples):
            point = X[i] #row in 2D array
            cluster = y[i]  #cluster assignment
            
            #compute distance within cluster
            same_cluster = X[y == cluster]  #rows in X with same cluster assignment
            if len(same_cluster) > 1:  
                a_i = np.mean(cdist([point], same_cluster, metric='euclidean')[0][1:])
            else:
                a_i = 0  #if it's the only point, a_i is 0

            #compute lowest mean distance between clusters
            b_i = float('inf')
            for other_cluster in unique_clusters:
                if other_cluster == cluster:
                    continue  #skip self cluster
                
                other_cluster_points = X[y == other_cluster]#rows in X with the same different cluster assignment
                if len(other_cluster_points) > 0:
                    avg_distance = np.mean(cdist([point], other_cluster_points, metric='euclidean'))
                    b_i = min(b_i, avg_distance)  #find the minimum average inter-cluster distance
            
            #compute final score
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouette_scores
        
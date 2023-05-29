import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, pairwise_distances


class KMeans(BaseEstimator):
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        self.labels_    = None


    def fit(self, X):
        centers = self._randomCenters(X)
        for iter in range(self.max_iter):
            self._updateLabels(X, centers)
            newCenters = self._updateCenters(X)
            if mean_squared_error(centers, newCenters) < self.tol: break
            centers = newCenters
        return self

    def _randomCenters(self, X):
        return X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]


    def _updateLabels(self, X, centers):
        self.labels_ = np.argmin(pairwise_distances(X, centers), axis=1)

    
    def _updateCenters(self, X):
        return np.concatenate([X[self.labels_ == k].mean(axis=0)[None, :] for k in range(self.n_clusters)], axis=0)
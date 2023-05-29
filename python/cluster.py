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
    


class DBSCAN(BaseEstimator):
    
    def __init__(self, eps=.5, min_samples=5):
        self.eps            = eps
        self.min_samples    = min_samples
        self.labels_        = None
        self.n_clusters_     = None

    
    def fit(self, X):
        # -2 - undefined, -1 - noise
        self.labels_ = np.zeros(X.shape[0], dtype=int) - 2
        self.n_clusters_ = 0
        for p in range(X.shape[0]):
            if self.labels_[p] != -2: continue
            self._makeCluster(X, p)
        return self

    def _makeCluster(self, X, pind):
        if not self._isCorePoint(X, pind):
            self.labels_[pind] = -1
            return

        self.n_clusters_ += 1
        cluster = [pind]
        self.labels_[pind] = self.n_clusters_ - 1
        while len(cluster) > 0:
            current = cluster.pop()
            if self._isCorePoint(X, current):
                neighbours = self._getNeighbours(X, current)
                cluster = list(neighbours[self.labels_[neighbours] < 0]) + cluster
                self.labels_[neighbours] = self.n_clusters_ - 1
            
            
    def _isCorePoint(self, X, pind):
        return len(self._getNeighbours(X, pind)) > self.min_samples


    def _getNeighbours(self, X, pind):
        return np.where(pairwise_distances(X[pind][None, :], X)[0] < self.eps)[0]
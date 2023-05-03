import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DecisionTreeRegressor(BaseEstimator):

    # [feature < threshold]
    class _Node:
        def __init__(self, feature, threshold, val=None, isLeaf=False):
            self.feature = feature
            self.threshold = threshold
            self.left = self.right = None
            self.val = val
            self.isLeaf = isLeaf

        def goToLeft(self, x):
            return x[self.feature] < self.threshold

    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2, 
                 min_impurity_decrease=0, criterion='squared_error'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.impurity = DecisionTreeRegressor._getImpurity(criterion)
        self.tree = None


    def _predict_by_object(self, x):
        if self.tree is None:
            raise 'The tree hasn\'t fitted yet' 

        node = self.tree
        while not node.isLeaf:
            if node.goToLeft(x):
                node = node.left
            else:
                node = node.right
        return node.val


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._predict_by_object(x) for x in X])


    def fit(self, X, y):
        Hm = self.impurity(y)
        self.tree = self.make_node(X, y, 0, Hm)
        return self


    def make_node(self, X, y, depth, Hm):
        if self.min_samples_split >= X.shape[0] or ((self.max_depth is not None) and (depth >= self.max_depth)):
            val = y.mean()
            return self._Node(feature=None, threshold=None, val=val, isLeaf=True)
        
        thresholds = self._getThresholds(y)
        theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit = None, None, None
        for feature in range(X.shape[1]):
            for threshold in thresholds:
                Rl, yl, Rr, yr = self._divide(X, y, feature, threshold)
                if Rl.shape[0] < self.min_samples_leaf or Rr.shape[0] < self.min_samples_leaf: continue
                Hl, Hr = self.impurity(yl), self.impurity(yr)
                impurityDecrease = Hm - Rl.shape[0] / X.shape[0] * Hl - Rr.shape[0] / X.shape[0] * Hr
                if theBestImpurityDecrease is None or impurityDecrease > theBestImpurityDecrease:
                    theBestImpurityDecrease = impurityDecrease
                    theBestThreshold = threshold
                    theBestFeatureSplit = feature

        if theBestImpurityDecrease < self.min_impurity_decrease:
            val = y.mean()
            return self._Node(feature=None, threshold=None, val=val, isLeaf=True)
        
        Rl, yl, Rr, yr = self._divide(X, y, theBestFeatureSplit, theBestThreshold)
        node = self._Node(theBestFeatureSplit, theBestThreshold, None, False)
        Hl, Hr = self.impurity(yl), self.impurity(yr)
        node.left = self.make_node(Rl, yl, depth + 1, Hl)
        node.right = self.make_node(Rr, yr, depth + 1, Hr)
        return node


    def _divide(self, X, y, feature, threshold):
        mask = X[:, feature] < threshold
        return X[mask], y[mask], X[~mask], y[~mask]


    def _getThresholds(self, y):
        thresholds = np.unique(y)
        thresholds.sort()
        return (thresholds[:-1] + thresholds[:-1]) / 2

    def _getImpurity(criterion):
        return squared_error



def squared_error(y):
    return ((y - y.mean()) ** 2).mean()
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

    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=1,
                 min_impurity_decrease=0, criterion='squared_error'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.impurity = DecisionTreeRegressor._getImpurity(criterion)
        self.tree = None
        self._max_depth = None

    def _predict_by_object(self, x):
        if self.tree is None:
            raise Exception('The tree hasn\'t fitted yet')

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
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.values
        
        Hm = self.impurity(y)
        self.tree = self.make_node(X, y, 0, Hm)
        return self


    def make_node(self, X, y, depth, Hm):
        if (self._max_depth is None) or (depth > self._max_depth) or (Hm <= self.min_impurity_decrease):
            self._max_depth = depth
        
        if self.min_samples_split >= X.shape[0] or ((self.max_depth is not None) and (depth >= self.max_depth)):
            val = self._theBestValue(y)
            return self._Node(feature=None, threshold=None, val=val, isLeaf=True)

        theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit = self._theBestSplit(X, y, Hm)
        if (theBestImpurityDecrease is None) or (theBestImpurityDecrease <= self.min_impurity_decrease):
            val = self._theBestValue(y)
            return self._Node(feature=None, threshold=None, val=val, isLeaf=True)
        
        Rl, yl, Rr, yr = self._divide(X, y, theBestFeatureSplit, theBestThreshold)
        node = self._Node(theBestFeatureSplit, theBestThreshold, None, False)
        Hl, Hr = self.impurity(yl), self.impurity(yr)
        node.left = self.make_node(Rl, yl, depth + 1, Hl)
        node.right = self.make_node(Rr, yr, depth + 1, Hr)
        return node


    def _theBestSplit(self, X, y, Hm):
        theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit = None, None, None
        for feature in range(X.shape[1]):
            thresholds = self._getThresholds(X[:, feature])
            for threshold in thresholds:
                Rl, yl, Rr, yr = self._divide(X, y, feature, threshold)
                if Rl.shape[0] < self.min_samples_leaf or Rr.shape[0] < self.min_samples_leaf: continue
                Hl, Hr = self.impurity(yl), self.impurity(yr)
                impurityDecrease = Hm - Rl.shape[0] / X.shape[0] * Hl - Rr.shape[0] / X.shape[0] * Hr
                if theBestImpurityDecrease is None or impurityDecrease > theBestImpurityDecrease:
                    theBestImpurityDecrease = impurityDecrease
                    theBestThreshold = threshold
                    theBestFeatureSplit = feature
        return theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit


    def _divide(self, X, y, feature, threshold):
        mask = X[:, feature] < threshold
        return X[mask], y[mask], X[~mask], y[~mask]


    def _getThresholds(self, y):
        thresholds = np.unique(y)
        thresholds.sort()
        return (thresholds[:-1] + thresholds[1:]) / 2

    def _getImpurity(criterion):
        if criterion == "squared_error":    return squared_error
        if criterion == "gini":             return gini
        raise Exception('Unknown criterion! Criterion must be \'squared_error\'')


    def _theBestValue(self, y):
        return y.mean()



# classes must be integer numbers from 0....N - 1
class DecisionTreeClassifier(DecisionTreeRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nClasses = None

    def fit(self, X, y):
        self.nClasses = np.unique(y).size
        super().fit(X, y)
        return self

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def _theBestValue(self, y):
        res = np.zeros(self.nClasses)
        targets, counts = np.unique(y, return_counts=True)
        p = np.array([count / self.nClasses for count in counts])
        res[targets] = p
        return res


def gini(y):
    targets, counts = np.unique(y, return_counts=True)
    N = counts.sum()
    p = np.array([count / N for count in counts])
    return 1 - (p ** 2).sum()


def squared_error(y):
    return ((y - y.mean()) ** 2).mean()
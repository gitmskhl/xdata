import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DecisionTreeRegressor(BaseEstimator):

    # [feature < threshold]
    class _Node:
        def __init__(self, feature, threshold, num_objects, val=None, isLeaf=False):
            self.feature = feature
            self.threshold = threshold
            self.left = self.right = None
            self.num_objects = num_objects
            self.val = val
            self.isLeaf = isLeaf

        def goToLeft(self, x):
            if np.isnan(x[self.feature]): return None
            return x[self.feature] < self.threshold
        
        def print_tree(self, tab=0):
            if self.left is not None:
                print('\t' * tab, f'Node x[{self.feature}] < {self.threshold}')
                print('\t' * tab, ' Left: ')
                self.left.print_tree(tab + 1)
                print('\t' * tab, ' Right: ')
                self.right.print_tree(tab + 1)
            else:
                print('\t' * tab, f'Leaft val = {self.val}')
        
    class _CatNode(_Node):
        # threshold is a list of categorical values
        def __init__(self, feature, threshold, num_objects, val=None, isLeaf=False):
            super().__init__(feature, threshold, num_objects, val, isLeaf)
        
        def goToLeft(self, x):
            if np.isnan(x[self.feature]): return None
            return x[self.feature] in self.threshold
        
        def print_tree(self, tab=0):
            print('\t' * tab, f'Node x[{self.feature}] in {self.threshold}')
            print('\t' * tab, ' Left: ')
            self.left.print_tree(tab + 1)
            print('\t' * tab, ' Right: ')
            self.right.print_tree(tab + 1)
        


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
        self.cat_features = self.num_features = None


    # def _predict_by_object(self, x):
    #     node = self.tree
    #     while not node.isLeaf:
    #         goLeft = node.goToLeft(x)
    #         if goLeft is None:
    #             pass
    #         elif goLeft:
    #             node = node.left
    #         else:
    #             node = node.right
    #     return node.val

    def _predict_by_object(self, x, node):
        while not node.isLeaf:
            goLeft = node.goToLeft(x)
            if goLeft is None:
                left = self._evaluate(x, node.left)
                right = self._evaluate(x, node.right)
                return node.left.num_objects / node.num_objects * left + node.right.num_objects / node.num_objects * right
            elif goLeft:
                node = node.left
            else:
                node = node.right
        return node.val

    def predict(self, X):
        if self.tree is None:
            raise Exception('The tree hasn\'t fitted yet')
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._predict_by_object(x, self.tree) for x in X])


    def fit(self, X, y, cat_features=None):
        X = X.copy()
        y = y.copy()
        self.save_features(X, cat_features)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.values
        
        Hm = self.impurity(y)
        self.tree = self.make_node(X, y, 0, Hm)
        return self

    def save_features(self, X, cat_features):
        if cat_features is None: cat_features = np.array([])
        else:
            try:
                cat_features = np.array(cat_features, dtype=int)
            except:
                raise Exception('cat_features param must be a list of indices or None')
            if cat_features.ndim != 1:
                raise Exception('cat_features param must be a 1D array of indices')
            if np.unique(cat_features).size != cat_features.size:
                raise Exception('cat_features param must contain unique indices')
            if cat_features.size != 0 and (cat_features.min() < 0 or cat_features.max() >= X.shape[1]):
                raise Exception(f"cat_features param must contain indices in range from 0 to {X.shape[1] - 1}")
        
        self.cat_features = cat_features
        num_mask = ~np.isin(np.arange(X.shape[1], dtype=int), cat_features)
        self.num_features = np.arange(X.shape[1], dtype=int)[num_mask]
        self.cat_feature_values = [{} for _ in cat_features]


    def make_node(self, X, y, depth, Hm):
        if (self._max_depth is None) or (depth > self._max_depth) or (Hm <= self.min_impurity_decrease):
            self._max_depth = depth
        
        if self.min_samples_split >= X.shape[0] or ((self.max_depth is not None) and (depth >= self.max_depth)):
            val = self._theBestValue(y)
            return self._Node(feature=None, threshold=None, num_objects=X.shape[0], val=val, isLeaf=True)

        theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit, isCatFeature, theBestDefault = self._theBestSplit(X, y, Hm)
        if (theBestImpurityDecrease is None) or (theBestImpurityDecrease <= self.min_impurity_decrease):
            val = self._theBestValue(y)
            return self._Node(feature=None, threshold=None, num_objects=X.shape[0], val=val, isLeaf=True)
        
        if isCatFeature:
            Rl, yl, Rr, yr = self._divideCatFeature(X, y, theBestFeatureSplit, theBestThreshold, theBestDefault)
            node = self._CatNode(feature=theBestFeatureSplit, threshold=theBestThreshold, num_objects=X.shape[0], val=None, isLeaf=False)
        else:
            Rl, yl, Rr, yr = self._divideNumFeature(X, y, theBestFeatureSplit, theBestThreshold, theBestDefault)
            node = self._Node(feature=theBestFeatureSplit, threshold=theBestThreshold, num_objects=X.shape[0], val=None, isLeaf=False)
        
        Hl, Hr = self.impurity(yl), self.impurity(yr)
        node.left = self.make_node(Rl, yl, depth + 1, Hl)
        node.right = self.make_node(Rr, yr, depth + 1, Hr)
        return node


    def _theBestSplit(self, X, y, Hm):
        theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit, theBestDefault = None, None, None, None
        for feature in range(X.shape[1]):

            if feature in self.cat_features:
                impurityDecrease, threshold, isDefaultLeft = self.__theBestSplitByCatFeature(X, y, feature, Hm)
            else:
                impurityDecrease, threshold, isDefaultLeft = self.__theBestSplitByNumFeature(X, y, feature, Hm)
            
            if impurityDecrease is None: continue
            if theBestImpurityDecrease is None or impurityDecrease > theBestImpurityDecrease:
                theBestImpurityDecrease = impurityDecrease
                theBestThreshold = threshold
                theBestFeatureSplit = feature
                theBestDefault = isDefaultLeft
        
        isCatFeature = theBestFeatureSplit in self.cat_features
        return theBestImpurityDecrease, theBestThreshold, theBestFeatureSplit, isCatFeature, theBestDefault


    def __theBestSplitByCatFeature(self, X, y, feature, Hm):
        X_last = X
        X, y, missing, X_missing, y_missing = self.__missing(X, y, feature)

        def divide(X, y, preprocessed_cat_values, threshold):
            mask = preprocessed_cat_values < threshold
            return X[mask], y[mask], X[~mask], y[~mask]

        cat_unique, cat_preprocessed, preprocessed_cat_values = self._catPreproc(X[:, feature], y)
        theBestImpurityDecrease, theBestThreshold = None, None
        thresholds = self._getThresholds(cat_preprocessed)
        for threshold in thresholds:
            Rl, yl, Rr, yr = divide(X, y, preprocessed_cat_values, threshold)
            if Rl.shape[0] < self.min_samples_leaf or Rr.shape[0] < self.min_samples_leaf: 
                continue
            Hl, Hr = self.impurity(yl), self.impurity(yr)
            impurityDecrease = Hm - Rl.shape[0] / X.shape[0] * Hl - Rr.shape[0] / X.shape[0] * Hr
            if theBestImpurityDecrease is None or impurityDecrease > theBestImpurityDecrease:
                theBestImpurityDecrease = impurityDecrease
                theBestThreshold = cat_unique[cat_preprocessed < threshold]

        isDefaultLeft = None
        if missing:
            theBestImpurityDecrease, isDefaultLeft = self.__adjustedImpurityDecrease(X, y, feature, X_last, Hm, X_missing, y_missing, theBestThreshold)

        return theBestImpurityDecrease, theBestThreshold, isDefaultLeft


    def _catPreproc(self, cat_array, y):
        cat_unique = np.unique(cat_array)
        cat_preprocessed = np.array([y[cat_array == cat_val].mean() for cat_val in cat_unique])
        inds = np.argsort(cat_preprocessed)
        preprocessed_array = np.zeros_like(cat_array, dtype=np.float64)
        for i, cat_val in enumerate(cat_unique):
            preprocessed_array[cat_array == cat_val] = cat_preprocessed[i]
        return cat_unique[inds], cat_preprocessed[inds], preprocessed_array


    def __theBestSplitByNumFeature(self, X, y, feature, Hm):
        X_last = X
        X, y, missing, X_missing, y_missing = self.__missing(X, y, feature)
        
        theBestImpurityDecrease, theBestThreshold = None, None
        thresholds = self._getThresholds(X[:, feature])
        for threshold in thresholds:
            Rl, yl, Rr, yr = self._divideNumFeature(X, y, feature, threshold, None)
            if Rl.shape[0] < self.min_samples_leaf or Rr.shape[0] < self.min_samples_leaf: continue
            Hl, Hr = self.impurity(yl), self.impurity(yr)
            impurityDecrease = Hm - Rl.shape[0] / X.shape[0] * Hl - Rr.shape[0] / X.shape[0] * Hr
            if theBestImpurityDecrease is None or impurityDecrease > theBestImpurityDecrease:
                theBestImpurityDecrease = impurityDecrease
                theBestThreshold = threshold

        isDefaultLeft = None
        if missing:
            theBestImpurityDecrease, isDefaultLeft = self.__adjustedImpurityDecrease(X, y, feature, X_last, Hm, X_missing, y_missing, theBestThreshold)

        return theBestImpurityDecrease, theBestThreshold, isDefaultLeft


    def __adjustedImpurityDecrease(self, X, y, feature, X_last, Hm, X_missing, y_missing, theBestThreshold):
        Rl, yl, Rr, yr = self._divideNumFeature(X, y, feature, theBestThreshold, None)
        Hl, Hr = self.impurity(np.concatenate([yl, y_missing])), self.impurity(yr)
        impurityDecreaseL = Hm - (Rl.shape[0] + X_missing.shape[0]) / X_last.shape[0] * Hl - Rr.shape[0] / X_last.shape[0] * Hr

        Hl, Hr = self.impurity(yl), self.impurity(np.concatenate([yr, y_missing]))
        impurityDecreaseR = Hm - Rl.shape[0] / X_last.shape[0] * Hl - (Rr.shape[0] + X_missing.shape[0]) / X_last.shape[0] * Hr
        
        theBestImpurityDecrease = np.max(impurityDecreaseL, impurityDecreaseR)
        isDefaultLeft = theBestImpurityDecrease == impurityDecreaseL
        return theBestImpurityDecrease, isDefaultLeft


    def __missing(self, X, y, feature):
        mask = np.isnan(X[:, feature])
        missing = np.any(mask)
        if missing:
            X_missing, y_missing = X[mask], y[mask]
            X, y = X[~mask], y[~mask]
        if missing:
            return X, y, missing, X_missing, y_missing
        else:
            return X, y, missing, None, None

    def _divideCatFeature(self, X, y, feature, threshold, isDefaultLeft):
        X, y, missing, X_missing, y_missing = self.__missing(X, y, feature)
        cat_array = X[:, feature]
        mask = np.isin(cat_array, threshold)
        X_left, y_left, X_right, y_right = X[mask], y[mask], X[~mask], y[~mask]
        if isDefaultLeft is not None:
            if isDefaultLeft:
                X_left, y_left = np.concatenate([X_left, X_missing], axis=0), np.concatenate([y_left, y_missing], axis=0)
            else:
                X_right, y_right = np.concatenate([X_right, X_missing], axis=0), np.concatenate([y_right, y_missing], axis=0)
        
        return X_left, y_left, X_right, y_right

    def _divideNumFeature(self, X, y, feature, threshold, isDefaultLeft):
        X, y, missing, X_missing, y_missing = self.__missing(X, y, feature)
        mask = X[:, feature] < threshold
        X_left, y_left, X_right, y_right = X[mask], y[mask], X[~mask], y[~mask]
        if isDefaultLeft is not None:
            if isDefaultLeft:
                X_left, y_left = np.concatenate([X_left, X_missing], axis=0), np.concatenate([y_left, y_missing], axis=0)
            else:
                X_right, y_right = np.concatenate([X_right, X_missing], axis=0), np.concatenate([y_right, y_missing], axis=0)
        
        return X_left, y_left, X_right, y_right


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


    def show(self):
        self.tree.print_tree()


# classes must be integer numbers from 0....N - 1
class DecisionTreeClassifier(DecisionTreeRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nClasses = None

    def fit(self, X, y, cat_features=None):
        self.nClasses = np.unique(y).size
        if (cat_features is not None) and self.nClasses > 2:
            raise Exception("Can use param 'cat_features' only for binary classification")
        super().fit(X, y, cat_features)
        return self

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def _theBestValue(self, y):
        res = np.zeros(self.nClasses)
        targets, counts = np.unique(y, return_counts=True)
        N = counts.sum()
        p = np.array([count / N for count in counts])
        res[targets] = p
        return res
    
    def _catPreproc(self, cat_array, y):
        cat_unique = np.unique(cat_array)
        cat_preprocessed = np.array([(y[cat_array == cat_val] == 1).mean() for cat_val in cat_unique])
        inds = np.argsort(cat_preprocessed)
        preprocessed_array = np.zeros_like(cat_array, dtype=np.float64)
        for i, cat_val in enumerate(cat_unique):
            preprocessed_array[cat_array == cat_val] = cat_preprocessed[i]
        return cat_unique[inds], cat_preprocessed[inds], preprocessed_array


def gini(y):
    targets, counts = np.unique(y, return_counts=True)
    N = counts.sum()
    p = np.array([count / N for count in counts])
    return 1 - (p ** 2).sum()


def squared_error(y):
    return ((y - y.mean()) ** 2).mean()
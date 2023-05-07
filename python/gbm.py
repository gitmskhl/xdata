import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class ConstRegressor(BaseEstimator):
    def __init__(self, val):
        self.val = val

    def fit(self, X, y):
        self.val = y.mean()

    def predict(self, X):
        return self.val

class MultRegressor(BaseEstimator):
    def __init__(self, baseModel, coef):
        self.baseModel = baseModel
        self.coef = coef

    def fit(self, X, y): pass

    def predict(self, X):
        return self.baseModel.predict(X) * self.coef

# L(y, z) = (y - z)^2
# L'(y, z) = 2 * (z - y)
class GBMRegressor(BaseEstimator):
    
    def __init__(self, n_estimators=100, base_estimator=DecisionTreeRegressor, learning_rate=.1,
                 loss=mean_squared_error, lossDeriv=lambda y, z: 2 * (z - y), save_loss=False, 
                 early_stopping_rounds=None, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = base_estimator
        self.base_models = None
        self.kwargs = kwargs
        self.loss = loss
        self.lossDeriv = lossDeriv
        self.save_loss = save_loss
        self.history_train, self.history_valid = [] if save_loss else None, [] if save_loss else None
        self.early_stopping_rounds = early_stopping_rounds

    def predict(self, X):
        if self.base_models is None:
            raise Exception("The GBM hasn't fitted yet")
        if isinstance(X, pd.DataFrame):
            X = X.values

        res = 0
        for base_model in self.base_models:
            res = res + base_model.predict(X) * self.learning_rate
        return res
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.base_models = [ConstRegressor(y.mean())]
        current_predicts = y.mean() * self.learning_rate
        if X_valid is not None:
            current_valid_predicts = y.mean() * self.learning_rate
        
        if self.early_stopping_rounds is not None:
            last_losses = np.full(self.early_stopping_rounds, np.inf)
            if X_valid is None:
                raise Exception('Validation sample needed')
        
        for _ in range(self.n_estimators):
            model = self.base_estimator(**self.kwargs)
            model.fit(X, -self.lossDeriv(y, current_predicts))
            new_predictions = model.predict(X)
            gamma = self._get_optimal_gamma(y, current_predicts, new_predictions)
            self.base_models.append(MultRegressor(model, gamma))
            current_predicts = current_predicts + gamma * self.learning_rate * new_predictions

            if X_valid is not None:
                current_valid_predicts = current_valid_predicts + gamma * self.learning_rate * model.predict(X_valid)

            if self.early_stopping_rounds is not None:
                last_losses[:-1] = last_losses[1:]
                last_losses[-1] = self.loss(y_valid, current_valid_predicts)
                if np.all(last_losses[:-1] <= last_losses[1:]): break

            if self.save_loss:
                self.history_train.append(self.loss(y, current_predicts))
                if X_valid is not None:
                    self.history_valid.append(self.loss(y_valid, current_valid_predicts))
                    

        return self
    

    def _get_optimal_gamma(self, y, last_predictions, new_predictions):
        gammas = np.linspace(0, 1, 100)
        losses = np.array([self.loss(y, last_predictions + gamma * new_predictions) for gamma in gammas])
        return gammas[np.argmin(losses)]
    

class MyXGBoostTree(BaseEstimator):

    class _Node:
        def __init__(self, feature, threshold, isLeaf, val=None, left=None, right=None):
            self.feature    = feature
            self.threshold  = threshold
            self.isLeaf     = isLeaf
            self.val        = val
            self.left       = left
            self.right      = right


        def goToLeft(self, x):
            return x[self.feature] <= self.threshold


        def print_tree(self, tab=0):
            if self.left is not None:
                print('\t' * tab, f'Node x[{self.feature}] < {self.threshold}')
                print('\t' * tab, ' Left: ')
                self.left.print_tree(tab + 1)
                print('\t' * tab, ' Right: ')
                self.right.print_tree(tab + 1)
            else:
                print('\t' * tab, f'Leaft val = {self.val}')


    class _Split:
        def __init__(self, feature, threshold, impurityDecrease, Xl, yl, Xr, yr, Sl, Hl, Sr, Hr, predictionsL, predictionsR):
            self.feature                            = feature
            self.threshold                          = threshold
            self.impurityDecrease                   = impurityDecrease
            self.Xl, self.yl                        = Xl, yl
            self.Xr, self.yr                        = Xr, yr
            self.Sl, self.Hl                        = Sl, Hl
            self.Sr, self.Hr                        = Sr, Hr
            self.predictionsL, self.predictionsR    = predictionsL, predictionsR


    def __init__(self, lossDeriv, loss2Deriv, gamma_, lambda_, max_depth):
        self.lossDeriv  = lossDeriv
        self.loss2Deriv =   loss2Deriv
        self.gamma_     = gamma_
        self.lambda_    = lambda_
        self.root       = None
        self.max_depth  = max_depth


    def _predict_by_object(self, x, node):
        while not node.isLeaf:
            goLeft = node.goToLeft(x)
            if goLeft:
                node = node.left
            else:
                node = node.right
        return node.val

    def predict(self, X):
        if self.root is None:
            raise Exception('The tree hasn\'t fitted yet')
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._predict_by_object(x, self.root) for x in X])
    


    def fit(self, X, y, current_predictions):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.values

        S, H = self.__getSH(y, current_predictions)
        self.root = self.make_node(X, y, current_predictions, S, H, 0)

        return self


    def make_node(self, Xm, ym, predictions_m, Sm, Hm, depth):
        if (self.max_depth is not None) and depth == self.max_depth:
            return self._make_leaf(Sm, Hm)

        theBestSplit = None
        for feature in range(Xm.shape[1]):
            theBestSplitByFeature = self._theBestSplitByFeature(feature, Xm, ym, predictions_m, Sm, Hm)
            if (theBestSplit is None) or (theBestSplitByFeature.impurityDecrease > theBestSplit.impurityDecrease):
                theBestSplit = theBestSplitByFeature
        
        if (theBestSplit is None) or (theBestSplit.impurityDecrease <= 0):
            return self._make_leaf(Sm, Hm)


        leftNode = self.make_node(theBestSplit.Xl, theBestSplit.yl, theBestSplit.predictionsL, theBestSplit.Sl, theBestSplit.Hl, depth=depth + 1)

        rightNode = self.make_node(theBestSplit.Xr, theBestSplit.yr, theBestSplit.predictionsR, theBestSplit.Sr, theBestSplit.Hr, depth=depth + 1)

        return self._Node(feature=theBestSplit.feature, threshold=theBestSplit.threshold, isLeaf=False, val=None, left=leftNode, right=rightNode)



    def _theBestSplitByFeature(self, feature, Xm, ym, predictions_m, Sm, Hm) -> _Split:
        impurity = self._impurity(Sm, Hm)
        inds = np.argsort(Xm[:, feature])
        Xm, ym, predictions_m = Xm[inds], ym[inds], predictions_m[inds]
        thresholds, indexes, counts = np.unique(Xm[:, feature], return_index=True, return_counts=True)
        theBestSplit = None
        Sl = Hl = 0
        for i, threshold in enumerate(thresholds[:-1]):
            inds_start, inds_end = indexes[i], indexes[i] + counts[i]
            Sl = Sl + self.__getS(ym[inds_start:inds_end], predictions_m[inds_start:inds_end])
            Hl = Hl + self.__getH(ym[inds_start:inds_end], predictions_m[inds_start:inds_end])
            Sr, Hr = Sm - Sl, Hm - Hl

            impurityDecrease = impurity - self._impurity(Sl, Hl) - self._impurity(Sr, Hr)
            if (theBestSplit is None) or (theBestSplit.impurityDecrease < impurityDecrease):
                Xl, yl = Xm[:inds_end], ym[:inds_end]
                Xr, yr = Xm[inds_end:], ym[inds_end:]
                predictionsL, predictionsR = predictions_m[:inds_end], predictions_m[inds_end:]
                theBestSplit = self._Split(feature, threshold, impurityDecrease, Xl, yl, Xr, yr, Sl, Hl, Sr, Hr, predictionsL, predictionsR)
        
        return theBestSplit


    def _theBestSplitByFeatureOptimized(self, feature, Xm, ym, predictions_m, Sm, Hm) -> _Split:
        pass


    def _make_leaf(self, S, H) -> _Node:
        return self._Node(feature=None, threshold=None, isLeaf=True, val=self._theBestValue(S, H))


    def __getSH(self, Y, current_predictions):
        return self.__getS(Y, current_predictions), self.__getH(Y, current_predictions)

    def __getH(self, Y, current_predictions):
        return self.__get_h(Y, current_predictions).sum()
        
    def __getS(self, Y, current_predictions):
        return self.__get_s(Y, current_predictions).sum()

    def __get_h(self, y, current_prediction):
        return self.loss2Deriv(y, current_prediction)
    
    def __get_s(self, y, current_prediction):
        return -self.lossDeriv(y, current_prediction)

    def _impurity(self, S, H):
        return - 1.0 * (S ** 2) / (2.0 * (H + self.lambda_)) + self.gamma_

    def _theBestValue(self, S, H):
        return 1.0 * S / (H + self.lambda_)


class MyXGBoost(BaseEstimator):
    
    
    def __init__(self, lossDeriv, loss2Deriv, n_estimators=100, gamma_=1, lambda_=1, learning_rate=.1, max_depth=None, save_history=False, metric=None):
        self.lossDeriv = lossDeriv
        self.loss2Deriv = loss2Deriv
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_models = None
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.save_history = save_history
        self.history = None if not save_history else []
        self.metric = metric
        self.max_depth = max_depth

    def predict(self, X):
        if self.base_models is None:
            raise Exception("MyXGBoost hasn't fitted yet")
        if isinstance(X, pd.DataFrame):
            X = X.values

        res = 0
        for base_model in self.base_models:
            res = res + base_model.predict(X) * self.learning_rate
        return res
    

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.base_models = [ConstRegressor(0)]
        current_predicts = np.full(y.size, 0) * self.learning_rate
        for _ in range(self.n_estimators):
            model = MyXGBoostTree(lossDeriv=self.lossDeriv, loss2Deriv=self.loss2Deriv, gamma_=self.gamma_, lambda_=self.lambda_, max_depth=self.max_depth)
            model.fit(X, y, current_predicts)
            current_predicts = current_predicts + self.learning_rate * model.predict(X)
            self.base_models.append(model)

            if self.save_history:
                self.history.append(self.metric(y, current_predicts))

        return self
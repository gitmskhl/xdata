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
        
        for i in range(self.n_estimators):
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
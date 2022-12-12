import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin


class kNNRegression(BaseEstimator, RegressorMixin):
    def __init__(self, neighbors):
        self.neighbors = neighbors
    
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def euclidean_distance(self, X_train, y_train, X_test, K):
        distances = [(y, d) for y, d in zip(y_train, np.sqrt(np.sum((X_train-X_test)**2, axis=1)))]
        return [y for y, _ in sorted(distances, key = lambda x: x[1])][:K]

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            predictions.append(
                np.mean(
                    self.euclidean_distance(self.X_train, self.y_train, X_test.iloc[i,:], self.neighbors)
                )
            )
        return predictions
    



train = pd.DataFrame(
    {
        'height': [100, 20, 30],
        'weight': [20, 60, 40],
        'target': [1, 2, 3]
    }
)

test = pd.DataFrame(
    {
        'height': [80, 10, 20],
        'weight': [10, 50, 30],
        'target': [0, 1, 2]
    }
)

def gridSearch(X, y, n_list, est_better, est_min, estimator = metrics.mean_squared_error) -> int:
        max_est = est_min
        curr = -1
        for parameter in n_list:
            reg = kNNRegression(neighbors=parameter)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
                )
            reg.fit(X_train, y_train)
            est = estimator(reg.predict(X_test), y_test)
            if est_better(est, max_est):
                curr = parameter
                max_est = est
        return curr
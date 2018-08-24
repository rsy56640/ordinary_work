from enum import Enum
import numpy as np


def square_loss(Pi_positive, lam, n_U, X_U, X_S):
    Pi_S = 2 * Pi_positive ^ 2 + 1 - 2 * Pi_positive
    A = (2 * Pi_positive - 1) / n_U * (X_U.T.dot(X_U) + 2 * lam * X_U * np.eye(X_U.shape[1]))
    b = 2 * Pi_S * X_S.T.mean(axis=1) - X_U.T.mean(axis=1)
    return np.linalg.solve(A, b)


class SU_Classification(object):
    class Prior_knowledge(Enum):
        exact_Pi = 1
        nothing = 2
        sign = 3

    def __init__(self, n_U: int, n_S: int, _lambda: float, prior_knowledge: Prior_knowledge, Pi: float,
                 Loss_Function=square_loss):
        if prior_knowledge is SU_Classification.Prior_knowledge.exact_Pi:
            self._Pi_positive = Pi
            self._Pi_negative = 1 - self._Pi_positive
        else:
            assert Pi >= 0.5, "Suppose Pi_positive is grater than Pi_negative"
            self._Pi_positive = (1 + np.sqrt(2 * Pi - 1)) / 2.0
            self._Pi_negative = 1 - self._Pi_positive
        self._Loss_Function = Loss_Function
        self._n_U = n_U
        self._n_S = n_S
        self._lambda = _lambda


def getParam(self):
    return self._Loss_Function(self._Pi_positive)

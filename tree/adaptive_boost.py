"""
adaptive boost classification tree
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    """
    adaptive boost classifier with decision stump
    """
    def __init__(self, m_sizes=20):
        self.m_sizes = m_sizes
        self.lst_clf = []
        self.lst_alpha = []

    def _compute_error(self, y_pred, y_true, w):
        # w are already normalizated when updated
        e_m = np.sum((y_pred == y_true) * w)
        if e_m == 0:
            print(y_pred)
            print(y_true)
            assert 0
        return e_m

    def _compute_alpha(self, error):
        return 1/2 * np.log((1 - error)/error)

    def _update_weights(self, y_pred, y_true, w_m, alpha_m):
        z_m = np.sum(np.exp(-1 * alpha_m * y_pred * y_true))
        w_m_1 = w_m/z_m * np.exp(-1 * alpha_m * y_pred * y_true)
        return w_m_1

    def fit(self, x_train, y_train):
        # pylint: disable=W1401
        """
        1. N samples training dataset $T = \{(x_1, y_1), (x_2, y_2),..., (x_N, y_N)\} x_i \in R^n and y_i \in \{-1,1\}$
        2. M base classifier, G_m(x), G_m(x) is decision stump, means it is a decision tree with max_depth = 1
        3. Weights for train samples in the m-th round iteration D_m = (w_m1, w_m2,..., w_mN) 
            and every element of  D_0 is initialized to 1 / N
        4. The classification error for G_m is defined as the following:
           $ e_m = \sum_{i=1}^N {P(G_m(x_i) \neq y_i)} = \sum_{i=1}^N {w_{mi}I(G_m(x_i) \neq y_i)} $
        5. calculate weight alpha for base classifier G_m
            $ \alpha_m = 1/2 *\log{\frac{1-e_m}{em}} $
        6. update weight for traning samples, this is a normalization update
            $ w_{m+1, i} = \frac{w_{mi}}{Z_m}\exp{(-\alpha_m*y_i*G_m(x_i))} $
            $ Z_m = \sum_{i=1}^N{\exp{(-\alpha_m*y_i*G_m(x_i))}} $
        7. the final classifer is 
            $ f(x) = \sum_{i=1}^M{\alpha_m*G_m(x_i)} $
            G(x) = sign(f(x))
        """
        # pylint: enable=W1401
        n_samples, _ = x_train.shape

        # init training sample weights
        w_m = np.ones(n_samples) / n_samples
        for _ in range(self.m_sizes):

            clf = DecisionTreeClassifier(max_depth=1)
            print(y_train)
            clf.fit(x_train, y_train, sample_weight=w_m)
            y_pred = clf.predict(x_train)
            self.lst_clf.append(clf)

            e_m = self._compute_error(y_pred, y_train, w_m)
            alpha_m = self._compute_alpha(e_m)
            w_m = self._update_weights(y_pred, y_train, w_m, alpha_m)
            self.lst_alpha.append(alpha_m)

        assert len(self.lst_alpha) == len(self.lst_clf)


    def predict(self, x_test):
        # n_samples, n_features = x_test.shape
        # m_size, n_samples
        preds = np.array([clf.predicate(x_test) for clf in self.cls_clf])
        # n_samples, m_size
        preds = np.swapaxes(preds, 0, 1)

        # n_samples
        preds = np.dot(preds, self.lst_alpha)

        return np.sign(preds)

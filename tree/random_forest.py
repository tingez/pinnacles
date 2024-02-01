"""
code for random forest
"""
import numpy as np
import numpy.typing as npt
from collections import Counter
from decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest class
    """
    def __init__(self, n_trees:int=10, max_depth:int=10, min_samples_split:int=2, n_features:int=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []


    def fit(self, x_train: npt.ArrayLike, y_train: npt.ArrayLike):
        n_samples, n_features = x_train.shape

        self.n_features = min(self.n_features, n_features) if self.n_features else n_features

        for _ in range(self.n_trees):
            resample_idxs = np.random.choice(n_samples, n_samples, replace=True)
            resample_x_train, resample_y_train = x_train[resample_idxs], y_train[resample_idxs]
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            tree.fit(resample_x_train, resample_y_train)
            self.trees.append(tree)

    def _most_common_label(self, labels):
        return Counter(labels).most_common(1)[0][0]


    def predict(self, x_test: npt.ArrayLike):
        # => n_trees, n_sample
        predications = np.array([tree.predict(x_test) for tree in self.trees])
        # => n_sample, n_trees
        tree_preds = np.swapaxes(predications, 0, 1)
        predications = np.array([self._most_common_label(preds) for preds in tree_preds])
        return predications


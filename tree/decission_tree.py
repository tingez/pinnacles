"""
test for decision tree
"""
import typer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter


app = typer.Typer(pretty_exceptions_show_locals=False)

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecissionTree:
    """
        Decission Tree Class
    """
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def _most_common_label(self, y):
        y_cnter = Counter(y)
        return y_cnter.most_common(1)[0][0]


    def _empirical_entropy(self, labels):
        labels_cnter = Counter(labels)
        total_cnt = len(labels)

        entropy = 0
        for _, value in labels_cnter.items():
            p = value / total_cnt
            entropy += - 1 * p * np.log(p)

        return entropy

    def _split(self, x_train, feature_id, threshold):
        feature_train = x_train[:, feature_id]

        l_idxs = np.argwhere(feature_train <= threshold).flatten()
        r_idxs = np.argwhere(feature_train > threshold).flatten()
        return l_idxs, r_idxs

    def _information_gain(self, x_train, y_train, feature_id, threshold):
        # g(D, A) = H(D) - H(D|A)

        # calculate H(D)
        parent_entropy = self._empirical_entropy(y_train)

        # calculate H(D|A)
        l_idxs, r_idxs = self._split(x_train, feature_id, threshold)

        l_entropy = self._empirical_entropy(y_train[l_idxs])
        r_entropy = self._empirical_entropy(y_train[r_idxs])

        child_entropy = l_entropy * len(l_idxs) / len(x_train) + r_entropy * len(r_idxs) / len(x_train)
        return parent_entropy - child_entropy

    def _best_split(self, x_train, y_train, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        #two loops for every feature and possible pivot
        for feat_idx in feature_idxs:
            x_train_feat = x_train[:, feat_idx]

            feature_thresholds = np.unique(x_train_feat)
            for thr in feature_thresholds:
                gain = self._information_gain(x_train, y_train, feat_idx, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _grow_tree(self, x_train, y_train, depth=0):
        n_samples, n_features = x_train.shape
        n_labels = len(np.unique(y_train))

        # terminate criterion
        if depth >= self.max_depth or n_samples <= self.min_sample_split or n_labels == 1:
            leaf_value = self._most_common_label(y_train)
            return TreeNode(value=leaf_value)

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(x_train, y_train, feature_idxs)

        # create left node and right node
        left_idxs, right_idxs = self._split(x_train, best_feature, best_threshold)
        left_node = self._grow_tree(x_train[left_idxs], y_train[left_idxs], depth+1)
        right_node = self._grow_tree(x_train[right_idxs], y_train[right_idxs], depth+1)

        return TreeNode(best_feature, best_threshold, left_node, right_node)


    def fit(self, x_train, y_train):
        _, n_features = x_train.shape
        self.n_features = min(self.n_features, n_features) if self.n_features else n_features
        self.root = self._grow_tree(x_train, y_train)


    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        feature_value = x[node.feature]
        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


    def predict(self, x_test):
        return np.array([self._traverse_tree(x, self.root) for x in x_test])


@app.command()
def build_dt():
    data = datasets.load_breast_cancer()

    x, y = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf = DecissionTree(max_depth=10)
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    acc = accuracy(y_test, predictions)
    print(acc)


if __name__ == '__main__':
    app()

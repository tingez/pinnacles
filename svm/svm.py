"""
test for svm
"""
import typer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

app = typer.Typer(pretty_exceptions_show_locals=False)

class SVM:
    """
    svm class
    """
    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, x_train, y_train):
        n_samples, n_dims = x_train.shape

        self.w = np.zeros(n_dims)
        self.b = 0

        y_sign = np.where(y_train<=0, -1, 1)
        # do stachastic gradient descent
        # y_pred = w * x + b
        # ridge regression Loss  = lambda||w||**2 + (1/n)*sum(max(0, 1 - y_sign * y_pred))
        # if y * y_linear >= 1
        # dw = 2 * lambda * w,  db = 0
        # else y * y_linear < 1
        # dw = 2 * lambda * w - (1/n) * sum(y_sign * x)
        # db = - (1/n) * sum(y_sign * 1)
        for _ in range(self.n_iters):
            for idx in range(n_samples):
                y_pred = np.dot(x_train[idx], self.w) + self.b

                condition = y_sign[idx] * y_pred >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w  - y_sign[idx] * x_train[idx]
                    db = - y_sign[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, x_test):
        approx = np.dot(x_test, self.w) - self.b
        return np.sign(approx)

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

@app.command()
def build_svm():
    # pylint: disable-next=W0632
    x, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y<=0, -1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    clf = SVM()
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)

    print('SVM classification accuracy', accuracy(y_test, predictions))

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)

        x0_1 = np.amin(x[:, 0])
        x0_2 = np.amax(x[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

        x1_min = np.amin(x[:, 1])
        x1_max = np.amax(x[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()


if __name__ == '__main__':
    app()

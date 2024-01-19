"""
test for logistic regression
"""
import typer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SkLogisticRegression


app = typer.Typer(pretty_exceptions_show_locals=False)

class LogisticRegression:
    """
    class for logistic regression
    """
    def __init__(self, lr=0.01, n_iters=1000, threshold=0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.t = threshold


    def fit(self, x, y):
        n_samples, n_dims = x.shape
        self.w = np.zeros(n_dims)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred =  1 / (1 + np.exp(-1 * np.dot(x, self.w) - self.b))

            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return

    def predict(self, x):
        y_preds =  1 / (1 + np.exp(-1 * np.dot(x, self.w) - self.b))
        return [1 if y_pred > self.t else 0 for y_pred in y_preds]


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

@app.command()
def build_lr():
    bc = datasets.load_breast_cancer()
    x, y = bc.data, bc.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)

    acc = accuracy(y_preds, y_test)
    print(acc)

@app.command()
def build_lr_sklearn():
    bc = datasets.load_breast_cancer()
    x, y = bc.data, bc.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf = SkLogisticRegression()
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)

    acc = accuracy(y_preds, y_test)
    print(acc)



if __name__ == '__main__':
    app()

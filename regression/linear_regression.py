"""
test for linear regression
"""
import typer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SkLinearRegression
import matplotlib.pyplot as plt


app = typer.Typer(pretty_exceptions_show_locals=False)

class LinearRegression:
    """
    class for linear regression
    """
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x, y):
        n_samples, n_dims = x.shape

        self.w = np.zeros(n_dims)
        self.b = 0
        print(f'w.shape: {self.w.shape}')

        for _ in range(self.n_iters):
            y_pred = np.dot(x, self.w) + self.b

            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, x):
        predictions = np.dot(x, self.w)  + self.b
        return predictions

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

@app.command()
def build_lr():
    # pylint: disable-next=W0632
    x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) #

    plt.figure(figsize=(8,6))
    plt.scatter(x[:, 0], y, color ='b', marker ='o', s =30)
    plt.show()

    reg = LinearRegression(lr=0.01)
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)

    mse_value = mse(y_test, predictions)
    print(mse_value)

    y_pred_line = reg.predict(x)
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(x, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()

@app.command()
def build_lr_sklearn():
    # pylint: disable-next=W0632
    x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    plt.figure(figsize=(8,6))
    plt.scatter(x[:, 0], y, color='b', marker='o', s=30)
    plt.show()

    reg = SkLinearRegression()
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)

    mse_value = mse(y_test, predictions)
    print(mse_value)

    y_pred_line = reg.predict(x)
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(x, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()

if __name__ == '__main__':
    app()

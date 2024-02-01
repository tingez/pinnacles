"""
 test code for tree relative alogrithm
"""
import typer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from random_forest import RandomForest
from decision_tree import DecisionTree

app = typer.Typer(pretty_exceptions_show_locals=False)

def accuarcy(y_pred, y_true):
    acc = sum(y_pred == y_true) / len(y_pred)
    return acc

@app.command()
def test_tree():
    x, y = datasets.load_breast_cancer(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf_rf = RandomForest(n_trees=20)
    clf_rf.fit(x_train, y_train)

    clf_dt = DecisionTree(max_depth=10)
    clf_dt.fit(x_train, y_train)

    y_rf_pred = clf_rf.predict(x_test)
    acc_rf = accuarcy(y_rf_pred, y_test)

    y_dt_pred = clf_dt.predict(x_test)
    acc_dt = accuarcy(y_dt_pred, y_test)
    print(f'RF acc is {acc_rf}, DT acc is {acc_dt}')

@app.command()
def test_random_forest():
    x, y = datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf = RandomForest(n_trees=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuarcy(y_pred, y_test)
    print(f'RandomForest acc: {acc}')

@app.command()
def test_decision_tree():
    x, y = datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf = DecisionTree(max_depth=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuarcy(y_pred, y_test)
    print(f'decision tree acc: {acc}')

if __name__ == '__main__':
    app()

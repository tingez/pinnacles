"""
data visualization for distribution
"""
import typer
#import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def display_binomial():
    _, ax = plt.subplots(1, 1)

    x  = [0, 1, 2, 3, 4, 5, 6]
    n, p = 6, 0.5

    rv = binom(n, p)

    ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1, label='Probability')
    ax.legend(loc='best', frameon=False)
    plt.show()



if __name__ == '__main__':
    app()

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

# to make this notebook's output stable across runs
np.random.seed(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



def knn_predict():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]
    rows = 60000
    #rows = 40000

    X_train, X_test, y_train, y_test = X[:rows], X[rows:], y[:rows], y[rows:]
    shuffle_index = np.random.permutation(rows)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    knn_clf = KNeighborsClassifier(n_jobs=2, n_neighbors=3, weights="uniform")
    #knn_clf.fit(X_train, y_train)
    param_grid = [
        {'n_neighbors': [2, 3, 4, 5], 'weights': ['uniform','distance']}
    ]

    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, n_jobs=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    print(grid_search.cv_results_)


    #cvs = cross_val_score(knn_clf, X_train, y_train, cv=3,scoring="accuracy",n_jobs=4)
    #print(cvs)


if __name__ == '__main__':
    knn_predict()
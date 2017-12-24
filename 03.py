# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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

    X_train, X_test, y_train, y_test = X[:rows], X[rows:], y[:rows], y[rows:]
    shuffle_index = np.random.permutation(rows)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # rows = 10000
    # X_train, y_train = X_train[:rows], y_train[:rows]

    # param_grid = [
    #     {'n_neighbors': [ 3,4,5], 'weights': ['distance']}
    # ]
    # #,scoring='accuracy'
    # grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=5, verbose=3)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)
    # print(grid_search.cv_results_)
    # print(grid_search.best_score_)
    #

    #
    # y_pred = grid_search.predict(X_test)
    # accuracy_score(y_test, y_pred)

    knn_clf = KNeighborsClassifier(n_jobs=4, n_neighbors=4, weights="distance")
    knn_clf.fit(X_train, y_train)
    # cvs = cross_val_score(knn_clf, X_train, y_train, cv=3,scoring="accuracy",n_jobs=4)
    # print(cvs)
    y_pred = knn_clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

def shift_knn_predict():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]
    rows = 60000

    X_train, X_test, y_train, y_test = X[:rows], X[rows:], y[:rows], y[rows:]


    X1_train = [shift_image(image,0,1) for image in X_train]
    X2_train = [shift_image(image,0,-1) for image in X_train]
    X3_train = [shift_image(image,1,0) for image in X_train]
    X4_train = [shift_image(image,-1,0) for image in X_train]

    X_train = np.r_[X_train,X1_train,X2_train,X3_train,X4_train]
    y_train = np.r_[y_train,y_train,y_train,y_train,y_train]

    shuffle_index = np.random.permutation(rows*5)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=4,  weights="distance")
    knn_clf.fit(X_train, y_train)
    # cvs = cross_val_score(knn_clf, X_train, y_train, cv=3,scoring="accuracy",n_jobs=4)
    # print(cvs)
    y_pred = knn_clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)


def titanic_predict():
    titanic_path = os.path.join("datasets","titanic","train.csv")
    data = pd.read_csv(titanic_path)
    data = data.drop(labels=["Name"], axis=1)
    X_train = data.drop(labels=["Survived","Embarked","Sex","Ticket","Cabin"], axis = 1)
    y_train = data["Survived"]


    #sgd_clf = SGDClassifier(random_state=42)
    #sgd_clf.fit(X_train, y_train)


if __name__ == '__main__':
    titanic_predict()
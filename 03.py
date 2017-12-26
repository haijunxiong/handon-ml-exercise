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
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from CategoricalEncoder import CategoricalEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

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


# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


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

def print_prcm(clf,X_train,y_train):

    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
    cm = confusion_matrix(y_train, y_train_pred)

    print(cm)
    print(precision_score(y_train, y_train_pred))
    print(recall_score(y_train, y_train_pred) )

def get_titanic_data(file_name):
    titanic_path = os.path.join("datasets", "titanic", file_name)
    data = pd.read_csv(titanic_path)

#    data_num = data.drop(labels=["PassengerId", "Name", "Survived", "Embarked", "Sex", "Ticket", "Cabin"], axis=1)
#   y_train = data["Survived"].copy()

    data["Embarked"].fillna(value='U', inplace=True)
    data["Sex"].fillna(value='unknow', inplace=True)
    data["Cabin"].fillna(value='unknow', inplace=True)
    #data["Fare_cat"] = np.ceil(data["Fare"] / 50)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["Pclass"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    X_data = strat_train_set.drop("Survived", axis=1)
    y_data = strat_train_set["Survived"].copy()

    X_test = strat_test_set.drop("Survived", axis=1)
    y_test = strat_test_set["Survived"].copy()


    #print(list(data))
    data_num = strat_train_set[["Fare","Parch","Age"]]


    num_attribs = list(data_num)
    print(num_attribs)
    cat_attribs = ["Pclass", "Embarked", "Sex"]


    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    X_data_prepared = full_pipeline.fit_transform(X_data)
    X_test_prepared = full_pipeline.transform(X_test)

    return X_data_prepared, y_data,X_test_prepared,y_test

def print_accuracy_score(clf,X_test,y_test):
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)


def titanic_predict():


    #print(data_prepared)
    X_train,y_train,X_test, y_test = get_titanic_data("train.csv")
    #X_test, y_test = get_titanic_data("test.csv")

    # sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(X_train, y_train)
    # cvs = cross_val_score(sgd_clf, data_prepared, y_train, cv=3, scoring="accuracy")
    # print(cvs)
    #
    # y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
    # cm = confusion_matrix(y_train, y_train_pred)
    #
    # print(cm)
    # print(precision_score(y_train, y_train_pred))
    # print(recall_score(y_train, y_train_pred) )
    #
    # knn_clf = KNeighborsClassifier()
    # knn_clf.fit(X_train, y_train)
    #
    # y_train_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
    # cm = confusion_matrix(y_train, y_train_pred)
    #
    # print(cm)
    #
    # clf = knn_clf
    # print(precision_score(y_train, y_train_pred))
    # print(recall_score(y_train, y_train_pred) )

    forest_reg = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_leaf=2, n_jobs=-1,random_state=42)
    forest_reg.fit(X_train, y_train)
    print_prcm(forest_reg, X_train, y_train)
    print('test')
    print_prcm(forest_reg, X_test, y_test)
    print_accuracy_score(forest_reg,X_test,y_test)

    parameter_space = {
        "n_estimators": [5,10,20,50],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1,2, 4, 6,8,10,12,14,16,18,20],
    }

    scores = ['precision', 'recall', 'roc_auc']
    scores = ['accuracy']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(clf, parameter_space, cv=3, n_jobs =-1, scoring='%s' % score)
        # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
        grid.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print(grid.best_score_)
        print_prcm(grid.best_estimator_,X_train,y_train)

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



    # voting_clf = VotingClassifier(
    #     estimators=[('forest_reg', forest_reg), ('knn_clf', knn_clf), ('sgd_clf', sgd_clf)],
    #     voting='hard'
    # )
    # voting_clf.fit(X_train, y_train)
    #
    # clf = voting_clf
    # y_train_pred = cross_val_predict(voting_clf, X_train, y_train, cv=3)
    # cm = confusion_matrix(y_train, y_train_pred)
    #
    # print(cm)
    # print(precision_score(y_train, y_train_pred))
    # print(recall_score(y_train, y_train_pred) )



    # y_probas_forest = cross_val_predict(clf, X_train, y_train, cv=3, method="predict_proba")
    # y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
    # fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)
    # print(fpr_forest, tpr_forest, thresholds_forest )
    # print(roc_auc_score(y_train, y_scores_forest))



if __name__ == '__main__':
    titanic_predict()
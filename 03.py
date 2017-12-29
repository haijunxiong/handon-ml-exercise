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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

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
        return X[self.attribute_names]


# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)

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

    #data["Embarked"].fillna(value='U', inplace=True)
    #data["Sex"].fillna(value='unknow', inplace=True)
    #data["Cabin"].fillna(value='unknow', inplace=True)

    data["CabinCat"] = data["Cabin"].isnull()
    data["AgeBucket"] = data["Age"] // 15 * 15
    data["RelativesOnboard"] = data["SibSp"] + data["Parch"]

    #data["Fare_cat"] = np.ceil(data["Fare"] / 50)

    # split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # for train_index, test_index in split.split(data, data["Pclass"]):
    #     strat_train_set = data.loc[train_index]
    #     strat_test_set = data.loc[test_index]

    if  data.columns.contains('Survived'):
        X_data = data.drop("Survived", axis=1)
        y_data = data["Survived"].copy()
    else:
        X_data = data
        y_data = None

    #X_test = strat_test_set.drop("Survived", axis=1)
    #y_test = strat_test_set["Survived"].copy()


    #print(list(data))
    #data_num = data[["Fare", "Parch", "Age", "SibSp", "RelativesOnboard"]]
    #cat_attribs = ["Pclass","Sex",  "Embarked","AgeBucket"]

    data_num = data[["Fare","Parch","RelativesOnboard","Age", "SibSp"]]

    num_attribs = list(data_num)
    print(num_attribs)
    cat_attribs = ["Pclass","CabinCat","Sex", "Embarked"]


    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        #("PolynomialFeatures",PolynomialFeatures(degree=4, include_bias=False)),
        #('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('cat_selector', DataFrameSelector(cat_attribs)),
        ("imputer", MostFrequentImputer()),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    X_data_prepared = full_pipeline.fit_transform(X_data)
   # X_test_prepared = full_pipeline.transform(X_test)

    return X_data_prepared, y_data,data

def print_accuracy_score(clf,X_test,y_test):
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

def output_test_data(X_test,y_pred):
    df = pd.DataFrame(data={'PassengerId': X_test["PassengerId"],'Survived': y_pred})
    df.to_csv("d:/titanic.csv",columns=["PassengerId","Survived"],index=False)


def grid_search_clf(clf, parameter_space, X_train, y_train, score='accuracy'):
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid = GridSearchCV(clf, parameter_space, cv=3, n_jobs=-1, scoring='%s' % score)
    grid.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print(grid.best_score_)

    print_score(grid.best_estimator_, X_train, y_train)


def titanic_sgd():
    X_train, y_train, X_data = get_titanic_data("train.csv")
    sgd_clf = SGDClassifier(random_state=42)

    parameter_space = {
        "loss": ['hinge', 'log','modified_huber', 'squared_hinge' ,'perceptron']
    }

    grid_search_clf(sgd_clf,parameter_space,X_train,y_train)


def titanic_knn():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
        "n_neighbors": [1,2,3,5,6,7],
        "weights" : ['uniform','distance']
    }

    grid_search_clf(KNeighborsClassifier(), parameter_space, X_train,y_train)


def titanic_forest():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
        "n_estimators": [80,90,100,110],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [3, 4,5, 6],
        "max_leaf_nodes" : [-1,2,3],
        "max_features" : [4,5,6,7,8,9,10,11]
    }

    grid_search_clf(RandomForestClassifier(), parameter_space, X_train, y_train)

def titanic_svc():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000,1100,1200], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
        {'C': [1,3,5,10], 'degree' : [2,3,4,5],  'kernel' : ["poly"] }
    ]

    grid_search_clf(SVC(), parameter_space, X_train, y_train)

def titanic_bayes():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = [
    ]

    #grid_search_clf(GaussianNB(), parameter_space, X_train, y_train)

    clf =BernoulliNB()
    clf.fit(X_train,y_train)
    print_score(clf,X_train,y_train)


def print_score(clf,X_train, y_train):

    print(clf)
    print_accuracy_score(clf, X_train, y_train)

    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")
    print(scores)
    print(scores.mean())
    print()


def titanic_predict():

    X_train,y_train,X_data = get_titanic_data("train.csv")

    X_test, y_test, X_test_data = get_titanic_data("test.csv")

    # sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(X_train, y_train)
    # print_score(sgd_clf,X_train,y_train)

    svc_clf = SVC(**{'C': 3, 'degree': 3, 'kernel': 'poly','probability':True})
    svc_clf.fit(X_train, y_train)
    print_score(svc_clf,X_train,y_train)


    forest_reg = RandomForestClassifier(**{'n_estimators': 80, 'max_leaf_nodes': -1, 'criterion': 'entropy', 'max_features': 10, 'min_samples_leaf': 5}, random_state=42)
    forest_reg.fit(X_train, y_train)
    print_score(forest_reg, X_train, y_train)

    knn_clf = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    knn_clf.fit(X_train,y_train)
    print_score(knn_clf, X_train, y_train)

    voting_clf = VotingClassifier(
        estimators=[('forest_reg', forest_reg), ('knn_clf', knn_clf),('svc_clf',svc_clf)],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)

    clf = voting_clf
    print_score(clf, X_train, y_train)

    print('test')
    y_test_pred = clf.predict(X_test)
    output_test_data(X_test_data,y_test_pred)

if __name__ == '__main__':
    titanic_bayes()
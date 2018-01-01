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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):

        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


class TopFeatureSelector2(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices):
        self.feature_indices_ = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


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

    # shuffled_indices = np.random.permutation(len(data))
    # data = data.iloc[shuffled_indices]

#    data_num = data.drop(labels=["PassengerId", "Name", "Survived", "Embarked", "Sex", "Ticket", "Cabin"], axis=1)
#   y_train = data["Survived"].copy()

    #data["Embarked"].fillna(value='U', inplace=True)
    #data["Sex"].fillna(value='unknow', inplace=True)
    #data["Cabin"].fillna(value='unknow', inplace=True)

    data["CabinCat"] = data["Cabin"].str.get(0).fillna('N')
    data["AgeBucket"] = data["Age"] // 15 * 15
    data["RelativesOnboard"] = data["SibSp"] + data["Parch"]
    data["Title"] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    data["FirstName"] = data.Name.str.extract('([A-Za-z]+),', expand=False)

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


    num_attribs = ["Fare","Parch","RelativesOnboard","Age", "SibSp"]
    print(num_attribs)
    cat_attribs = ["Pclass","Sex", "Embarked"]
    cat_encoder1 = CategoricalEncoder(encoding="onehot-dense")

    #
    cat_encoder2 = CategoricalEncoder(categories=[['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'Dona']],encoding="onehot-dense")

    cat_encoder1.fit_transform(X_data[cat_attribs].dropna())
    cat_encoder2.fit_transform(X_data[['Title']].dropna())

    attributes = np.concatenate((np.array(num_attribs), np.concatenate(np.array(cat_encoder1.categories_))))
    attributes = np.concatenate((attributes, np.concatenate(np.array(cat_encoder2.categories_))))
    print(attributes)


    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('cat_selector', DataFrameSelector(cat_attribs)),
        ("imputer", MostFrequentImputer()),
        ('cat_encoder', cat_encoder1),
    ])

    cat_pipeline2 = Pipeline([
        ('cat_selector', DataFrameSelector(['Title'])),
        ("imputer", MostFrequentImputer()),
        ('cat_encoder', cat_encoder2),
    ])



    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("cat_pipeline2",cat_pipeline2)
    ])

    # cat_pipeline3 = Pipeline([
    #     ('cat_selector', DataFrameSelector(['Fare','female','male','Age',3,'RelativesOnboard','Master','Miss',1,'Mrs','Parch']))
    # ])



    prepare_select_and_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', TopFeatureSelector2([ 0,  2,  3,  4,  7,  8,  9, 26]))

    ])


    #X_data_prepared = full_pipeline.fit_transform(X_data)



    X_data_prepared = prepare_select_and_pipeline.fit_transform(X_data)

    return X_data_prepared, y_data,data

def print_accuracy_score(clf,X_test,y_test):
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

def output_test_data(X_test,y_pred):
    df = pd.DataFrame(data={'PassengerId': X_test["PassengerId"],'Survived': y_pred})
    titanic_path = os.path.join("datasets", "titanic", "gender_submission.csv")
    df.to_csv(titanic_path,columns=["PassengerId","Survived"],index=False)


def grid_search_clf(clf, parameter_space, X_train, y_train, score='accuracy'):
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid = GridSearchCV(clf, parameter_space, cv=10, n_jobs=-1, scoring='%s' % score)
    grid.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print(grid.best_score_)

    print_score(grid.best_estimator_, X_train, y_train)

    return grid.best_estimator_


def titanic_sgd():
    X_train, y_train, X_data = get_titanic_data("train.csv")
    sgd_clf = SGDClassifier(random_state=42)

    parameter_space = {
        "loss": ['hinge', 'log','modified_huber', 'squared_hinge' ,'perceptron']
    }

    return grid_search_clf(sgd_clf,parameter_space,X_train,y_train)


def titanic_gbrt():
    X_train, y_train, X_data = get_titanic_data("train.csv")
    clf = GradientBoostingClassifier(random_state=42)

    parameter_space = {
        "n_estimators": [1,5,10,100,200],
        "learning_rate": [0.1, 0.2, 0.5, 1.0],
        "max_depth": [1,2,3,4,5]
    }

    return grid_search_clf(clf, parameter_space,X_train,y_train)

def titanic_ada():
    X_train, y_train, X_data = get_titanic_data("train.csv")
    clf = AdaBoostClassifier(random_state=42)

    parameter_space = {
        "n_estimators": [1,5,10,100,200],
        "learning_rate": [0.1, 0.2, 0.5, 1.0,1.5]
    }

    return grid_search_clf(clf, parameter_space,X_train,y_train)


def titanic_knn():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
        "n_neighbors": [1,2,3,5,6,7,8,9,10],
        "weights" : ['uniform','distance']
    }

    return grid_search_clf(KNeighborsClassifier(), parameter_space, X_train,y_train)


def titanic_extra_tree():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
        "n_estimators": [3, 5,9,10, 20,30, 40,50,60,70,80],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1,2,3, 4,5, 6,7,8],
        #"max_leaf_nodes" : [-1,2,3],
        "max_features" : [2,3,5,6,7,8]
    }

    return  grid_search_clf(ExtraTreesClassifier(random_state=42), parameter_space, X_train, y_train)

def titanic_forest():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
        "n_estimators": [5,9,10, 20,30, 40,50,60,70,80],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1,2,3, 4,5, 6,7,8],
        #"max_leaf_nodes" : [-1,2,3],
        "max_features" : [2,3,5,6,7,8]
    }

    clf =  grid_search_clf(RandomForestClassifier(random_state=42), parameter_space, X_train, y_train)
    #
    # feature_importances = clf.feature_importances_
    #
    # sorted_attr = sorted(zip(feature_importances, attrs), reverse=True)
    # print(sorted_attr)
    # print( indices_of_top_k(feature_importances, 8))

    return clf


def titanic_svc():

    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = [
       #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [0.1, 0.5, 1, 5, 10, 100,110], 'gamma': [ 0.4, 0.3, 0.2, 0.1,  0.02, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
        #{'C': [1,3,5], 'degree' : [2,3],  'kernel' : ["poly"] }
    ]

    return grid_search_clf(SVC(random_state=42, probability=True), parameter_space, X_train, y_train)


def titanic_bayes():
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
        'binarize': [0, 0.25, 0.5, 0.75,1],
        #'alpha ': [0.0, 1.0]
    }
    return grid_search_clf(BernoulliNB(), parameter_space, X_train, y_train)

def titanic_voting(forest_reg,knn_clf,svc_clf,bayes_clf,ada_clf):
    X_train, y_train, X_data = get_titanic_data("train.csv")

    parameter_space = {
         'weights' : [[1,1,1,1,1], [3, 1, 2, 1, 1], [2, 1, 2, 1, 1],[4, 1, 2, 1, 1], [2, 1, 1, 1, 1]],
         'voting' : ['soft','hard']
    }
    clf = VotingClassifier(
        estimators=[('forest_reg', forest_reg), ('knn_clf', knn_clf), ('svc_clf', svc_clf), ('bayes_clf', bayes_clf),
                    ('ada_clf', ada_clf)],
        # estimators=[('forest_reg', forest_reg),   ('svc_clf', svc_clf) ,   ('ada_clf', ada_clf)],

    )

    return grid_search_clf(clf, parameter_space, X_train, y_train)


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

    print(X_train.shape)
    print(X_test.shape)

    # sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(X_train, y_train)
    # print_score(sgd_clf,X_train,y_train)
    #**{'C': 3, 'degree': 3, 'kernel': 'poly', 'probability': True},

    voting_clf = titanic_voting(titanic_forest(), titanic_extra_tree(), titanic_svc(), titanic_gbrt(), titanic_ada())

    clf = voting_clf
    #print_score(clf, X_train, y_train)

    print('test')
    y_test_pred = clf.predict(X_test)
    output_test_data(X_test_data,y_test_pred)


if __name__ == '__main__':
    titanic_predict()
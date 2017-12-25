# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import os

import numpy as np
import pandas as pd

# to make this notebook's output stable across runs


import matplotlib
import matplotlib.pyplot as plt

import os
import tarfile
from six.moves import urllib

from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from CategoricalEncoder import CategoricalEncoder

np.random.seed(42)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()




def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values    


fetch_housing_data()
housing = load_housing_data()

# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
housing_cat = housing["ocean_proximity"]


num_attribs = list(housing_num)
#num_attribs = ["median_income"]
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
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

housing_prepared = full_pipeline.fit_transform(housing)
#housing_prepared
#housing_prepared.shape

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

"""
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(housing_prepared, housing_labels).predict(housing_prepared)
y_lin = svr_lin.fit(housing_prepared, housing_labels).predict(housing_prepared)
y_poly = svr_poly.fit(housing_prepared, housing_labels).predict(housing_prepared)



svr_rbf_mse = mean_squared_error(housing_labels, y_rbf)
svr_rbf_rmse = np.sqrt(svr_rbf_mse)
print("svr_rbf_rmse:",svr_rbf_rmse)

svr_lin_mse = mean_squared_error(housing_labels, y_lin)
svr_lin_rmse = np.sqrt(svr_lin_mse)
print("svr_lin_rmse:",svr_lin_rmse)

svr_poly_mse = mean_squared_error(housing_labels, y_poly)
svr_poly_rmse = np.sqrt(svr_poly_mse)
print("svr_poly_rmse:",svr_poly_rmse)

from sklearn.model_selection import GridSearchCV

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

svr_reg = SVR()
grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_svr = best_model.predict(housing_prepared)

svr_mse = mean_squared_error(housing_labels, y_svr)
svr_rmse = np.sqrt(svr_mse)
print("svr_rmse:",svr_rmse)

"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


if __name__=='__main__':
    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,n_jobs=5,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)

    rnd_search_model = rnd_search.best_estimator_

    rnd_search_predictions = rnd_search_model.predict(housing_prepared)

    rnd_search_mse = mean_squared_error(housing_labels, rnd_search_predictions)
    rnd_search_rmse = np.sqrt(rnd_search_mse)

    print(rnd_search.best_params_)
    print(rnd_search_rmse)

    feature_importances = rnd_search.best_estimator_.feature_importances_
    
    print(feature_importances)

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]


    cat_encoder = CategoricalEncoder()
    housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)

    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs

    print(attributes)
    sorted_attr=sorted(zip(feature_importances, attributes), reverse=True)
    print(sorted_attr)
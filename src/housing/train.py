import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from logger_main import configure_logger

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    
    # Use argparser to get the user command line input such as where to store data and pickle model

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest="dataset", type=str, default='./data/raw/housing.csv',
                        help='Path to dataset')
    parser.add_argument('--model-output', dest="model_output", type=str, default='./artifacts',
                        help='Output folder path for trained model')
    parser.add_argument('--log-level', dest="log_level", type=str, default='DEBUG',
                        help='Specify log level. e.g. `--log-level DEBUG, default is DEBUG')
    parser.add_argument('--log-path', dest="log_path", type=str, default=False,
                        help='use a log file or not. if yes give path,e.g. `--log-path <path>,default is not log file')
    parser.add_argument('--no-console-log', dest="log_console", action="store_true",
                        help='toggle whether or not to write logs to the console')
    args = parser.parse_args()

    # Based on the user input from the terminal, set the file path and console output true/false

    if args.log_path is not False:
        log_file_path = os.path.join(args.log_path, 'log_file.log')
    else:
        log_file_path = None
    if args.log_console is True:
        console_input = False
    else:
        console_input = True
    
    # Configure the logger based on the user input from the terminal

    configure_logger(logger=None, cfg=None, log_file=log_file_path, console=console_input, log_level=args.log_level)

    housing = pd.read_csv(args.dataset)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    # visualize the housing data using lat and long
    
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # Correlation matrix for housing
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    # select the numerical only data
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_num = housing.drop('ocean_proximity', axis=1)
    
    # custom transformer for the new features mentioned above

    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    attr_adder = CombinedAttributesAdder()
    housing_extra_attribs = attr_adder.transform(housing.values)    
    housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
                                         columns=list(housing.columns) 
                                         + ["rooms_per_household", "population_per_household", "bedrooms_per_room"],
                                         index=housing.index)
    # pipe line to impute,add new attribute and sclar transfromation

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                             ('attribs_adder', CombinedAttributesAdder()), 
                             ('std_scaler', StandardScaler())])
    
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    # drop labels for training set
    housing = strat_train_set.drop("median_house_value", axis=1)  
    housing_labels = strat_train_set["median_house_value"].copy()
    # column transformer to combine both num and cat columns
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),
                                       ("cat", OneHotEncoder(), cat_attribs),])
           
    # final test data and lables for training are

    housing_prepared = full_pipeline.fit_transform(housing)
    housing_labels = strat_train_set["median_house_value"].copy()

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    # random forest regressor with randomized search method
    forest_reg = RandomForestRegressor(random_state=42)
    param_distribs = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8)}

    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)

    # random forest regressor with grid search method
    param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
                  ]
    forest_reg = RandomForestRegressor(random_state=42)
    
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    feature_importances = grid_search.best_estimator_.feature_importances_

    sorted(feature_importances, reverse=True)

    # store the models in to the user defined/artifacts folder

    def store_model(model_dir):
        """
        Function to store the model files in to the user defined folder

        parameters
        ----------
        model_dir: directory path
            Directory where models will bestored

        returns
        ------
        Following models are generated,
            linear regression
            tree regression
            random serachCV
            gridsearchCV

        notes
        -----
        models are stored in .pkl extension [pickle files]
        
        """
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'lin_reg.pkl'), 'wb') as f:
            pickle.dump(lin_reg, f)
        with open(os.path.join(model_dir, 'tree_reg.pkl'), 'wb') as f:
            pickle.dump(tree_reg, f)
        with open(os.path.join(model_dir, 'rnd_search.pkl'), 'wb') as f:
            pickle.dump(rnd_search, f)
        with open(os.path.join(model_dir, 'grid_search.pkl'), 'wb') as f:
            pickle.dump(grid_search, f)
    # call the model function to generate and store the models
    store_model(args.model_output)

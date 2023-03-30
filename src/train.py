import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
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

    # Based on the user input from the terminal set the file path and console output true/false

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

    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()

    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop('ocean_proximity', axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[['ocean_proximity']]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    forest_reg = RandomForestRegressor(random_state=42)
    param_distribs = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8)}
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    # store the models in to the user defined/artifacts folder

    def store_model(model_dir):
        """
        Function to store the model files in to the user defined folder

        parameters
        ----------
        model_dir: directory path
            Directory where models will bestored

        yields
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

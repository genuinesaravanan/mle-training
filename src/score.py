import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from logger_main import configure_logger

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # get the user inputs from terminal

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', dest='model_path', type=str, default='./artifacts/',
                        help='Path to trained model')
    parser.add_argument('--dataset-path', dest='dataset_path', type=str, default='./data/raw/housing.csv',
                        help='Path to dataset for scoring')
    parser.add_argument('--output-path', dest='output_path', type=str, default='./artifacts/',
                        help='Output folder path for scoring results')
    parser.add_argument('--log-level', dest="log_level", type=str, default='DEBUG',
                        help='Specify log level. e.g. `--log-level DEBUG, default is DEBUG')
    parser.add_argument('--log-path', dest="log_path", type=str, default=False,
                        help='use a log file or not. if yes give path,e.g. `--log-path <path>,default is not log file')
    parser.add_argument('--no-console-log', dest="log_console", action="store_true",
                        help='toggle whether or not to write logs to the console')
    args = parser.parse_args()
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

    # Split the dataset in to train and test

    housing = pd.read_csv(args.dataset_path)
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

    # From the arg.model_path get the model list and store the scores into arg.output_path folder
    
    def score_models(model_dir, score_dir):
        """
        This function read the model files from the user defined directory, generates the score for models and 
        stores the score in user defined directory

        parameters
        ----------

        model_dir : str
            where the models are located, the files should be in .pkl format

        score_dir : str
            The directory where the scores files will be stored

        yields
        ------

        This function reads four models (linear regression, tree regression, random search CV and grid searchCV )
        and generates the scores using the validation datasets. The scores are stored in the txt files to the directory 
        given by the user

        """
        
        with open(os.path.join(args.model_path, 'lin_reg.pkl'), 'rb') as f:
            lin_reg = pickle.load(f)
            housing_predictions = lin_reg.predict(housing_prepared)
            lin_mse = mean_squared_error(housing_labels, housing_predictions)
            lin_rmse = np.sqrt(lin_mse)
            with open(os.path.join(args.output_path, 'lin_reg.txt'), 'a') as f:
                f.write("MSE={}, RMSE={}\n".format(lin_mse, lin_rmse))
            f.close()
        with open(os.path.join(args.model_path, 'tree_reg.pkl'), 'rb') as f:
            tree_reg = pickle.load(f)
            housing_predictions = tree_reg.predict(housing_prepared)
            tree_mse = mean_squared_error(housing_labels, housing_predictions)
            tree_rmse = np.sqrt(tree_mse)
            with open(os.path.join(args.output_path, 'tree_reg.txt'), 'a') as f:
                f.write("MSE={}, RMSE={}\n".format(tree_mse, tree_rmse))
            f.close()
        with open(os.path.join(args.model_path, 'rnd_search.pkl'), 'rb') as f:
            rnd_search = pickle.load(f)
            cvres = rnd_search.cv_results_
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                with open(os.path.join(args.output_path, 'rnd_search.txt'), 'a') as f:
                    f.write("RMSE={}, Parameters={}\n".format(np.sqrt(-mean_score), params))
            f.close()
        with open(os.path.join(args.model_path, 'grid_search.pkl'), 'rb') as f:
            grid_search = pickle.load(f)
            cvres = grid_search.cv_results_
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                with open(os.path.join(args.output_path, 'grid_search.txt'), 'a') as f:
                    f.write("RMSE={}, Parameters={}\n".format(np.sqrt(-mean_score), params))
            f.close()
    # call the score function and generate the score

    score_models(args.model_path, args.output_path)

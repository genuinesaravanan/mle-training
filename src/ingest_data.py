import argparse
import logging
import os
import tarfile
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from logger_main import configure_logger

logger = logging.getLogger(__name__)


def download_data(output_folder):
    """ Function to download the data from the url and store it in to the user defined
    folder.

    parameters
    ----------
    output_folder : str
        The directory name where the data has to be stored (eg. /data/raw)
        by default it will be stored in  /data/raw directory

    yields
    ------
    Downloads the data from url and stores it in to the user defined/default 
    directory.

    """
    url_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
    url = url_root + "datasets/housing/housing.tgz"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "housing.tgz")
    urllib.request.urlretrieve(url, file_path)
    housing_tgz = tarfile.open(file_path)
    housing_tgz.extractall(path=output_folder)
    housing_tgz.close()
    csv_path = os.path.join(output_folder, "housing.csv")
    return pd.read_csv(csv_path)


def split_data(data):
    """
    Function to split the raw data for test and validation, it uses the scikit learn
    train_test_split function.

    parameters 
    ----------
    data : str
        The data should be in pandas dataframe format

    notes
    -----
    test_size=0.2
    random_state=42

    yields
    ------
    This function splits the dataframe in to test.csv and val.csv and stores in data directory

    """
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, val_data


if __name__ == '__main__':

    # Get the directory to store the downloaded data

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./data/raw',
                        help='Output folder path')
    parser.add_argument('--log-level', dest="log_level", type=str, default='DEBUG',
                        help='Specify log level. e.g. `--log-level DEBUG, default is DEBUG')
    parser.add_argument('--log-path', dest="log_path", type=str, default=False,
                        help='use a log file or not. if yes give path,e.g. `--log-path <path>,default is not log file')
    parser.add_argument('--no-console-log', dest="log_console", action="store_true",
                        help='toggle whether or not to write logs to the console')
    args = parser.parse_args()
    
    if args.log_path is not False:
        log_file_path = os.path.join(args.log_path, 'log_file.log')
        logger.debug("log will be stored in the log file 'log_file.log'")
    else:
        log_file_path = None
    if args.log_console is True:
        console_input = False
        logger.debug("Setting the console log false")
    else:
        console_input = True
    
    # Configure the logger based on the user input from the terminal
    
    configure_logger(logger=None, cfg=None, log_file=log_file_path, console=console_input, log_level=args.log_level)
    
    # Download and split the data from URL

    data = download_data(args.output)
    logger.debug("File was successfully downloaded in to the folder")
    train_data, val_data = split_data(data)

    # Split the data for both training and validation

    train_file = os.path.join(args.output, 'train.csv')
    val_file = os.path.join(args.output, 'val.csv')
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    logger.debug("Test and validation file was successfully stored in to the folder")

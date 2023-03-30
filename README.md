# Median housing value prediction

This libarary is used to predict the median house values. Various models such as linear regression, tree regression, grid serach and random search mothods are employed and perfromance scores are also recorded. Based on the perfromance score better model is selected for the prediction.

Three scripts avilable in src directory

1. ingest_data.py - Downloads the data from the url https://raw.githubusercontent.com/ageron/handson-ml/master/. and splits the data for test and     validation

2. train.py -  reads the data from the downloaded folder and trains the models and stores it in to the models folder

3. Score.py - loads the model from model folder, then predict the prices based on validation data set and then stores the scores (MSE, RMSE) in to the output folder.

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## logging
 - Logging options are included in the scripts for debugging. The inputs for logger configuration will be collected from terminal.
 - default log level will be 'debug', if you want to change the log level use --log-level 'log mode' eg. --log-level info
 - logging will not be stored in to the log file, if you want to store the logging in to log file, pass the log path in the terminal 
    eg. --log-path /log
 - The log messgages will be displayed in console while running the scripts. if you dont want log messages then disable it 
    using the following flag --no-console-log

## To excute the script

- The repo contains the dependency file which is located in deploy/conda/linux_cpu_py37.yml. create a environment using this file with a environment name.  
- activate the environment
- clone the repo and then log in to the root directory
- then run the python files in order, first ingest_data.py then train.py followed by score.py (eg. python src/ingest_data.py)
- The scores and models are stored in artifacts folder.
- to buid the sphinx doc, log in to doc folder then run make html. 

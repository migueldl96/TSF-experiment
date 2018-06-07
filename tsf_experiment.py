#!/usr/bin/env python
# -*- coding: utf8 -*-

from tsf.windows import *
from tsf.pipeline import TSFPipeline
from tsf.grid_search import TSFGridSearch

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, make_scorer

from reporter import TSFReporter

from metrics import *
import random
import numpy as np
import pandas as pd
import inspect

from sacred import Experiment

from scipy.stats.mstats import gmean

import sys
reload(sys)
sys.setdefaultencoding('utf8')

# Experiment object
ex = Experiment()

# Scoring functions
SCORERS = {
    # 'name': [function, greater_is_better]
    'amae': [amae, False],
    'gm': [gm, True],
    'mae': [mae, False],
    'mse': [mean_squared_error, False],
    'mmae': [mmae, False],
    'ms': [ms, True],
    'mze': [mze, False],
    'tkendall': [tkendall, False],
    'wkappa': [wkappa, True],
    'spearman': [spearman, True]
}

# Models
MODELS = {
    # Regressors
    'LassoCV': LassoCV,
    'MLPRegressor': MLPRegressor,
    'SVR': SVR,
    'DecisionTreeRegressor': DecisionTreeRegressor,

    # Classifiers
    'RandomForestClassifier': RandomForestClassifier,
    'MLPClassifier': MLPClassifier,
    'LogisticRegression': LogisticRegression,
    'GaussianProcessClassifier': GaussianProcessClassifier,
    'SVC': SVC,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'GaussianNB': GaussianNB,
    'GradientBoostingClassifier': GradientBoostingClassifier

}


def read_data(files, train_path, test_path):
    # train
    train = []
    for file in files:
        path = train_path + file
        single_serie = pd.read_csv(path, header=None).values
        single_serie = single_serie.reshape(1, single_serie.shape[0])

        if len(train) == 0:
            train = single_serie
        else:
            train = np.append(train, single_serie, axis=0)

    # test
    test = []
    for file in files:
        path = test_path + file
        single_serie = pd.read_csv(path, header=None).values
        single_serie = single_serie.reshape(1, single_serie.shape[0])

        if len(test) == 0:
            test = single_serie
        else:
            test = np.append(test, single_serie, axis=0)

    return train, test


def create_pipe(pipe_steps, estimator, seed):
    steps = []
    if pipe_steps['cc']:
        steps.append(("cc", ClassChange(n_jobs=-1)))
    if pipe_steps['dw']:
        steps.append(("dw", DinamicWindow(n_jobs=-1)))
    if pipe_steps['ar']:
        steps.append(("ar", SimpleAR(n_jobs=-1)))
    if pipe_steps['model']:
        model = MODELS[estimator]
        if 'random_state' in inspect.getargspec(model.__init__)[0]:
            steps.append(("model", MODELS[estimator](random_state=seed)))
        else:
            steps.append(("model", MODELS[estimator]()))

    return TSFPipeline(steps)


def get_params(pipe_steps, tsf_config, model_config):
    params = []
    TSFBaseTransformer.horizon = tsf_config['horizon']
    ex.info['horizon'] = tsf_config['horizon']

    if pipe_steps['ar']:
        params.append(tsf_config['ar'])
    if pipe_steps['dw']:
        params.append(tsf_config['dw'])
    if pipe_steps['cc']:
        params.append(tsf_config['cc'])
    if pipe_steps['model']:
        params.append(model_config['params'])

    return params


def set_best_params(best_config):
    best_params = {}
    # SimpleAR
    best_params.update({'ar': {}})
    [best_params['ar'].update({key: value}) for key, value in best_config.iteritems() if 'ar__' in key.lower()]
    # DinamicWindow
    best_params.update({'dw': {}})
    [best_params['dw'].update({key: value}) for key, value in best_config.iteritems() if 'dw__' in key.lower()]
    # ClassChange
    best_params.update({'cc': {}})
    [best_params['cc'].update({key: value}) for key, value in best_config.iteritems() if 'cc__' in key.lower()]
    # Model
    best_params.update({'model': {}})
    [best_params['model'].update({key[7:]: value}) for key, value in best_config.iteritems() if 'model__' in key.lower()]

    ex.info['best_params'] = best_params

@ex.config
def configuration():
    seed = 1

    # files where time series are located
    reports_path = "reports/"
    train_path = "data/train"
    test_path = "data/test"
    files = ["RVR.txt", "humidity.txt", "windDir.txt", "windSpeed.txt", "QNH.txt", "temp.txt"]

    # steps of the model
    pipe_steps = {
        'ar': True,     # Standard autorregresive model using fixed window
        'dw': True,     # Dinamic Window based on stat limit
        'cc': True,    # Dinamic Window based on class change (classification oriented)
        'model': True   # Final estimator used for forecasting
    }

    # parameters of the windows model
    tsf_config = {
        'horizon': 1,   # Forecast distance
        'ar': {         # Standard autorregresive parameters
            'ar__n_prev': [1, 2, 3, 4, 5, 6, 7, 8]         # Number of previous samples to use
        },
        'dw': {         # Dinamic Window based on stat limit parameters
            'dw__stat': ['variance'],                   # Stat to calculate window size
            'dw__metrics': [['mean', 'variance']],      # Stats to resume information of window
            'dw__ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],        # Stat ratio to limit window size
            'dw__indexs': [None]                        # Indexs of series to be used
        },
        'cc': {         # Dinamic window based on class change parameters
            'cc__metrics': [['mean', 'variance']],      # Stats to resume information of window
            'cc__indexs': [None],                       # Indexs of series to be used
        },
    }

    # parameters of the estimator model
    model_config = {

    }

    run_id = -1


@ex.automain
def main(files, pipe_steps, tsf_config, model_config, seed, run_id, train_path, test_path, reports_path):

    ex.info['run_id'] = run_id
    reporter = TSFReporter(run_id, reports_path)

    # Read the data
    train, test = read_data(files, train_path, test_path)
    ex.info['endog'] = files[0]
    if len(files) > 1:
        ex.info['exogs'] = files[1:]

    # Set the seed
    ex.info['seed'] = seed
    np.random.seed(seed)
    random.seed(seed)

    # Create pipe and set the config
    pipe = create_pipe(pipe_steps, model_config['estimator'], seed)
    params = get_params(pipe_steps, tsf_config, model_config)

    # Create and fit TSFGridSearch
    scorer = SCORERS[model_config['scoring']]
    scoring = make_scorer(scorer[0], greater_is_better=scorer[1])
    gs = TSFGridSearch(pipe, params, scoring=scoring)
    gs.fit(X=[], y=train)
    set_best_params(gs.best_params_)
    print gs.best_params_

    # Predict using Pipeline
    predicted_train = gs.predict(train)
    predicted_test = gs.predict(test)

    # Offset series
    true_train = pipe.offset_y(train, predicted_train)
    true_test  = pipe.offset_y(test, predicted_test)
    reporter.set_train_series(true_train, predicted_train)
    reporter.set_test_series(true_test, predicted_test)

    # Performance
    reporter.set_task_type(model_config['type'])
    if model_config['type'] == 'regression':
        mse_train = mean_squared_error(true_train, predicted_train)
        mse_test = mean_squared_error(true_test, predicted_test)
        ex.info['performance'] = {"mse": {}}
        ex.info['performance']['mse'] = {"train": mse_train, "test": mse_test}

        print "MSE Train: " + str(mse_train)
        print "MSE Test: " + str(mse_test)

    elif model_config['type'] == 'classification':
        print true_train
        print predicted_train
        ccr_train = accuracy_score(true_train, predicted_train)
        ccr_test = accuracy_score(true_test, predicted_test)
        ex.info['performance'] = {"ccr": {}}
        ex.info['performance']['ccr'] = {"train": ccr_train, "test": ccr_test}
        cm = confusion_matrix(true_test, predicted_test)
        tprs = [float(cm[index, index])/np.sum(cm, axis=1)[index] for index in range(0, cm.shape[0])]
        ex.info['performance']['ms'] = min(tprs)
        ex.info['performance']['gm'] = gmean(tprs)

        print "CCR Train: " + str(ccr_train)
        print "CCR Test: " + str(ccr_test)
        print "GMS: " + str(gmean(tprs))
        print "MS: " + str(min(tprs))
        print "CM"
        print cm

        reporter.set_confusion_matrix(cm)

    reporter.save()

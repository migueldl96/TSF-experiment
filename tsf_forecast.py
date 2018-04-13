# -*- coding: utf8 -*-

from tsf.windows import *
from tsf.pipeline import TSFPipeline
from tsf.grid_search import TSFGridSearch

from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier

from reporter import TSFReporter

from metrics import *
import time
import random
import numpy as np
import pandas as pd

from sacred import Experiment
from sacred.observers import FileStorageObserver

from scipy.stats.mstats import gmean

import sys
reload(sys)
sys.setdefaultencoding('utf8')


def var_function(samples):
    return np.var(samples)


# RVR umbralizer
def umbralizer(sample):
    if sample < 1000:
        return 2
    elif 1000 < sample < 1990:
        return 1
    else:
        return 0


# Experiment object
ex = Experiment()
ex.observers.append(FileStorageObserver.create('reports'))

# Reporter object
reporter = TSFReporter()

# Scoring functions
SCORERS = {
    # 'name': [function, greater_is_better]
    'amae': [amae, False],
    'gm': [gm, True],
    'mae': [mae, False],
    'mmae': [mmae, False],
    'ms': [ms, True],
    'mze': [mze, False],
    'tkendall': [tkendall, False],
    'wkappa': [wkappa, True],
    'spearman': [spearman, True]
}

# Models
MODELS = {
    'LassoCV': LassoCV,
    'RandomForestClassifier': RandomForestClassifier,
    'MLPClassifier': MLPClassifier,
    'MLPRegressor': MLPRegressor
}


def read_data(files):
    data = []

    if not files:
        raise ValueError("There is no data. Please use -f to select a file to read.")

    for file in files:
        path = 'data/' + file
        single_serie = pd.read_csv(path, header=None).values
        single_serie = single_serie.reshape(1, single_serie.shape[0])

        if len(data) == 0:
            data = single_serie
        else:
            data = np.append(data, single_serie, axis=0)

    return data


def split_train_test(data, test_ratio):
    train_ratio = 1-test_ratio

    if len(data.shape) == 1:
        train_samples = int(len(data) * train_ratio)
        return data[:train_samples], data[train_samples:]
    else:
        train_samples = int(len(data[0]) * train_ratio)
        return data[:, :train_samples], data[:, train_samples:]


def create_pipe(pipe_steps, estimator, seed):
    steps = []
    if pipe_steps['cc']:
        steps.append(("cc", ClassChange()))
    if pipe_steps['dw']:
        steps.append(("dw", DinamicWindow()))
    if pipe_steps['ar']:
        steps.append(("ar", SimpleAR()))
    if pipe_steps['model']:
        steps.append(("model", MODELS[estimator](random_state=seed)))
    return TSFPipeline(steps)


def get_params(pipe_steps, tsf_config, model_config):
    params = []
    TSFBaseTransformer.horizon = tsf_config['horizon']
    ex.info['horizon'] = tsf_config['horizon']
    reporter.set_horizon(tsf_config['horizon'])
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
    best_params.update({'cc': {}})
    # Model
    best_params.update({'model': {}})
    [best_params['model'].update({key[7:]: value}) for key, value in best_config.iteritems() if 'model__' in key.lower()]

    ex.info['best_params'] = best_params

@ex.config
def configuration():
    seed = 0

    # ratio of samples for testing
    test_ratio = 0.3

    # files where time series are located
    files = ["temp.txt"]

    # steps of the model
    pipe_steps = {
        'ar': True,     # Standard autorregresive model using fixed window
        'dw': True,     # Dinamic Window based on stat limit
        'cc': False,    # Dinamic Window based on class change (classification oriented)
        'model': True   # Final estimator used for forecasting
    }

    # parameters of the windows model
    tsf_config = {
        'horizon': 1,   # Forecast distance
        'ar': {         # Standard autorregresive parameters
            'ar__n_prev': [1, 2]                        # Number of previous samples to use
        },
        'dw': {         # Dinamic Window based on stat limit parameters
            'dw__stat': ['variance'],                   # Stat to calculate window size
            'dw__metrics': [['mean', 'variance']],      # Stats to resume information of window
            'dw__ratio': [0.1],                         # Stat ratio to limit window size
            'dw__indexs': [None]                        # Indexs of series to be used
        },
        'cc': {         # Dinamic window based on class change parameters
            'cc__metrics': [['mean', 'variance']],      # Stats to resume information of window
            'cc__indexs': [None],                       # Indexs of series to be used
            'cc__umbralizer': [None]
        },
    }

    # parameters of the estimator model
    model_config = {
        'type': 'regression',
        'estimator': "LassoCV",
        'scoring': 'amae',
        'params': {

        }
    }


@ex.named_config
def rvr():
    files = ["RVR.txt", "temp.txt", "humidity.txt", "windDir.txt", "windSpeed.txt", "QNH.txt"]


@ex.automain
def main(files, test_ratio, pipe_steps, tsf_config, model_config, seed, _run):

    ex.info['run_id'] = _run._id

    start = time.time()

    # Read the data
    data = read_data(files)
    reporter.set_files(files)
    ex.info['endog'] = files[0]
    if len(files) > 1:
        ex.info['exogs'] = files[1:]

    # Set the seed
    ex.info['seed'] = seed
    np.random.seed(seed)
    random.seed(seed)
    reporter.set_seed(seed)

    # Umbralizer
    if model_config['type'] == "classification":
        data[0] = map(umbralizer, data[0])

    # Create pipe and set the config
    pipe = create_pipe(pipe_steps, model_config['estimator'], seed)
    reporter.set_pipe_steps(pipe_steps)
    params = get_params(pipe_steps, tsf_config, model_config)
    reporter.set_param_grid(tsf_config)
    reporter.set_model_config(model_config, pipe.steps[-1][1].__class__.__name__)

    # Split
    train, test = split_train_test(data, test_ratio)
    reporter.set_test_ratio(test_ratio)

    # Create and fit TSFGridSearch
    scorer = SCORERS[model_config['scoring']]
    scoring = make_scorer(scorer[0], greater_is_better=scorer[1])
    gs = TSFGridSearch(pipe, params, scoring=scoring)
    gs.fit(X=[], y=train)
    set_best_params(gs.best_params_)
    reporter.set_best_configuration(gs.best_params_)

    # Predict using Pipeline
    predicted_train = gs.predict(train)
    predicted_test = gs.predict(test)

    # Offset series
    true_train = pipe.offset_y(train, predicted_train)
    true_test  = pipe.offset_y(test, predicted_test)
    reporter.set_train_series(true_train, predicted_train)
    reporter.set_test_series(true_test, predicted_test)

    # Performance
    if model_config['type'] == 'regression':
        mse_train = mean_squared_error(true_train, predicted_train)
        mse_test = mean_squared_error(true_test, predicted_test)
        ex.info['performance'] = {"mse": {}}
        ex.info['performance']['mse'] = {"train": mse_train, "test": mse_test}

        reporter.set_mse_performance(mse_test, mse_train)

    elif model_config['type'] == 'classification':
        ccr_train = accuracy_score(true_train, predicted_train)
        ccr_test = accuracy_score(true_test, predicted_test)
        ex.info['performance'] = {"ccr": {}}
        ex.info['performance']['ccr'] = {"train": ccr_train, "test": ccr_test}
        cm = confusion_matrix(true_test, predicted_test)
        tprs = [float(cm[index, index])/np.sum(cm, axis=1)[index] for index in range(0, cm.shape[0])]
        ex.info['performance']['ms'] = min(tprs)
        ex.info['performance']['gm'] = gmean(tprs)

        reporter.set_ccr_performance(ccr_test, ccr_train)
        reporter.set_sensitivity(tprs)
        reporter.set_confusion_matrix(cm)

    end = time.time()

    reporter.set_exec_time(end-start)
    reporter.save()


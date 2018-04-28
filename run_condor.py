import htcondor
import json
import os

def generate_configs_files(models, windows, seeds, files, windows_config):
    id = 0
    if not os.path.exists('config/tmp/'):
        os.makedirs('config/tmp/')
    for model in models:
        # Dont apply DinamicWindow to endogenous if classification
        if model['type'] == "classification":
            windows_config['dw']['dw__indexs'] = [[1, 2, 3, 4]]
        for ar_window in windows['ar']:
            for dw_window in windows['dw']:
                for cc_window in windows['cc']:
                    # Ignore config file if no windows will be applied
                    if ar_window or dw_window or cc_window:
                        for seed in seeds:
                            with open("config/pid/config%d.json" % id, 'w') as f:
                                json.dump({
                                    "model_config": model,
                                    "pipe_steps": {
                                        "ar": ar_window,
                                        "dw": dw_window,
                                        "cc": cc_window
                                    },
                                    "seed": seed,
                                    "files": files[model['type']],
                                    "tsf_config": windows_config
                                }, f)
                            id = id+1

def condor():
    # Experiment configuration
    # Database
    files = {
        "classification": ["RVR.txt", "temp.txt", "humidity.txt", "windDir.txt", "windSpeed.txt", "QNH.txt"],
        "regression": ["temp.txt", "humidity.txt", "windDir.txt", "windSpeed.txt", "QNH.txt"]
    }

    # Models
    models = [
        {
            "estimator": "LogisticRegression",
            "type": "classification",
            "scoring": "amae",
            "params": {
                "penalty": ["l1", "l2"],
                "C": [0.5, 1.0, 1.5, 2.0]
            }
        },
        {
            "estimator": "MLPClassifier",
            "type": "classification",
            "scoring": "amae",
            "params": {
              "hidden_layer_sizes": [80, 90, 100, 110, 120],
              "momentum": [0.8, 0.85, 0.9]
            }
        },
        {
            "estimator": "RandomForestClassifier",
            "type": "classification",
            "scoring": "amae",
            "params": {
                "n_estimators": [7, 8, 9, 10, 11, 12],
                "max_features": ["auto", "log2", 0.2],
                "class_weight": ["balanced", "balanced_subsample"]
            }
        },
        {
            "estimator": "MLPRegressor",
            "type": "regression",
            "scoring": "mse",
            "params": {
                "hidden_layer_sizes": [80, 90, 100, 110, 120],
                "momentum": [0.8, 0.85, 0.9]
            }
        },
        {
            "estimator": "LassoCV",
            "type": "regression",
            "scoring": "mse",
            "params": {
                "n_alphas": [80, 90, 100, 110, 120]
            }
        }
    ]

    # Windows
    windows = {
        "ar": [True, False],
        "dw": [True, False],
        "cc": [True, False]
    }

    # Windows config grid
    windows_config = {
        'horizon': 1,
        'ar': {
            'ar__n_prev': [2, 3, 4, 5, 6, 7, 8]
        },
        'dw': {
            'dw__stat': ['variance'],
            'dw__metrics': [['mean', 'variance']],
            'dw__ratio': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            'dw__indexs': [[0, 1, 2, 3, 4, 5]]
        },
        'cc': {
            'cc__metrics': [['mean', 'variance']],
        },
    }

    # Seeds
    seeds = [10, 20, 30, 40, 50]

    # Generate temp config files
    generate_configs_files(models, windows, seeds, files, windows_config)

    # Parameters
    num_seeds = len(seeds)
    num_classification = len([model for model in models if model['type'] == 'classification'])
    num_regression = len([model for model in models if model['type'] == 'regression'])
    num_windows_comb = min(len(windows['ar']) * len(windows['dw']) * len(windows['cc']), 7)

    num_models = num_classification + num_regression
    num_process = num_seeds * num_windows_comb * num_models

    # Schedd object: Jobs execution manager
    schedd = htcondor.Schedd()

    # Sub object:
    sub = htcondor.Submit()

    # Submit info
    # Single experiment files configuration
    sub['seed'] = '($(ProcId) % ' + str(num_seeds) + ') + 1'
    sub['window'] = '($(ProcId) / ' + str(num_seeds) + ') % ' + str(num_windows_comb)
    sub['model'] = '$(ProcId) / ' + str(num_models * num_windows_comb)

    # Common configuration
    sub['cmd'] = '/usr/bin/env'
    sub['arguments'] = 'python tsf_experiment.py with config/tmp/config$(ProcId).json ' \
                       'run_id=$(ProcId) ' \
                       '-F reports/report_$(ProcId)_model$INT(model)_comb$INT(window)_seed$INT(seed)'
    sub['getenv'] = "True"
    sub['output'] = 'condor/outputs/output$(ProcId).out'
    sub['error'] = 'condor/errors/error$(ProcId).out'
    sub['log'] = 'condor/logs/log$(ProcId).out'

    # Queue subs
    with schedd.transaction() as txn:
        sub.queue(txn, num_process)


if __name__ == "__main__":
    condor()

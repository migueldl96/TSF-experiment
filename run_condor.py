# -*- coding: utf8 -*-

import htcondor
import json
import os
import click


def generate_configs_files(models, windows, seeds, files, windows_config, outputs):
    id = 0
    if not os.path.exists('config/.tmp/'):
        os.makedirs('config/.tmp/')
    for model in models:
        # Dont apply DinamicWindow to endogenous if classification
        for ar_window in windows['ar']:
            for dw_window in windows['dw']:
                for cc_window in windows['cc']:
                    # Ignore config file if no windows will be applied
                    if ar_window or dw_window or cc_window:
                        for seed in seeds:
                            with open("config/.tmp/config%d.json" % id, 'w') as f:
                                json.dump({
                                    "model_config": model,
                                    "pipe_steps": {
                                        "ar": ar_window,
                                        "dw": dw_window,
                                        "cc": cc_window
                                    },
                                    "seed":  seed,
                                    "files": files[model['type']],
                                    "train_path": files["train_path"],
                                    "test_path":  files["test_path"],
                                    "reports_path": outputs["reports"],
                                    "tsf_config": windows_config
                                }, f)
                            id = id+1

@click.command()
@click.option('--config_file', '-c', default=None, required=True,
              help=u'Fichero JSON con la configuración del experimento.')
def condor(config_file):
    # Experiment configuration
    try:
        f = open(config_file)
        ex_config = json.load(f)

    except IOError:
        raise IOError("File '%s' does not exist." % config_file)

    # Generate temp config files
    generate_configs_files(ex_config['models'], ex_config['windows'], ex_config['seeds'], ex_config['files'], ex_config['windows_config'], ex_config['outputs'])

    # Parameters
    num_seeds = len(ex_config['seeds'])
    num_classification = len([model for model in ex_config['models'] if model['type'] == 'classification'])
    num_regression = len([model for model in ex_config['models'] if model['type'] == 'regression'])
    num_windows_comb = min(len(ex_config['windows']['ar']) * len(ex_config['windows']['dw']) * len(ex_config['windows']['cc']), 7)

    num_models = num_classification + num_regression
    num_process = num_seeds * num_windows_comb * num_models

    # Output paths
    if not os.path.exists(ex_config["outputs"]["condor"]):
        os.makedirs(ex_config["outputs"]["condor"])
        os.makedirs(ex_config["outputs"]["condor"] + '/outputs')
        os.makedirs(ex_config["outputs"]["condor"] + '/errors')
        os.makedirs(ex_config["outputs"]["condor"] + '/logs')


    # Schedd object: Jobs execution manager
    schedd = htcondor.Schedd()

    # Sub object:
    sub = htcondor.Submit()

    # Submit info
    # Single experiment files configuration
    sub['seed'] = '($(ProcId) % ' + str(num_seeds) + ') + 1'
    sub['window'] = '($(ProcId) / ' + str(num_seeds) + ') % ' + str(num_windows_comb)
    sub['model'] = '$(ProcId) / ' + str(num_seeds * num_windows_comb)

    # Common configuration
    sub['cmd'] = '/usr/bin/env'
    sub['arguments'] = 'python tsf_experiment.py with config/.tmp/config$(ProcId).json ' \
                       'run_id=$(ProcId) ' \
                       '-F' + str(ex_config["outputs"]["reports"]) + '/report_$(ProcId)_model$INT(model)_comb$INT(window)_seed$INT(seed)'
    sub['getenv'] = "True"
    sub['output'] = str(ex_config["outputs"]["condor"]) + '/outputs/output$(ProcId).out'
    sub['error'] = str(ex_config["outputs"]["condor"]) + '/errors/error$(ProcId).out'
    sub['log'] = str(ex_config["outputs"]["condor"]) + '/logs/log$(ProcId).out'

    # Queue subs
    with schedd.transaction() as txn:
        sub.queue(txn, num_process)


if __name__ == "__main__":
    condor()

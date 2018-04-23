import htcondor


def condor():
    # Parameters
    num_process = 150

    # Schedd object: Jobs execution manager
    schedd = htcondor.Schedd()

    # Sub object:
    sub = htcondor.Submit()

    # Submit info
    # Single experiment files configuration
    sub['seed'] = '$(ProcId) % 5'
    sub['window'] = '$(ProcId) / 5'
    sub['windows_comb'] = '$(window) % 6'
    sub['model'] = '$(ProcId) / 30'

    # Common configuration
    sub['cmd'] = '/usr/bin/env'
    sub['arguments'] = 'python tsf_experiment.py with ' \
                       'config/models/model$INT(model).json ' \
                       'config/windows/comb$INT(windows_comb).json ' \
                       'seed=$INT(seed)'
    sub['getenv'] = "True"
    sub['output'] = 'condor/outputs/output$(ProcId).out'
    sub['error'] = 'condor/errors/error$(ProcId).out'
    sub['log'] = 'condor/logs/log$(ProcId).out'

    # Queue subs
    with schedd.transaction() as txn:
        sub.queue(txn, num_process)


if __name__ == "__main__":
    condor()

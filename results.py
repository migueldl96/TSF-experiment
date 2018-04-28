import os, glob
from mako.template import Template
import json
import numpy as np
import dateutil.parser

result_path = 'results/'
seeds = [10, 20, 30, 40, 50]

def init_result_dict():
    results = {
        'ccr': {
            'test': 0,
            'train': 0
        },
        'mse': {
            'test': 0,
            'train': 0
        },
        'ms': 0,
        'gm': 0
    }
    return results


def is_classification(model_id):
    return model_id < 3


def save_results():
    for model in range(0, 5):
        for comb in range(0, 6):
            # Experiment result path
            single_result_path = result_path + 'model' + str(model) + '_comb' + str(comb)

            if not os.path.exists(single_result_path):
                os.makedirs(single_result_path)

            # Results from seeds
            mean_results = init_result_dict()
            std_results = init_result_dict()

            train_mses = np.zeros(5)
            test_mses = np.zeros(5)
            train_ccrs = np.zeros(5)
            test_ccrs = np.zeros(5)
            gms = np.zeros(5)
            mss = np.zeros(5)

            times = np.zeros(5)

            for index, seed in enumerate(seeds):
                single_report_path = glob.glob('./REP/report_*_model' + str(model) + '_comb' + str(comb) + '_seed' + str(seed) + '/1')[0]
                info_f = single_report_path + '/info.json'
                run_f = single_report_path + '/run.json'

                json_info_file = open(info_f)
                json_info_data = json.load(json_info_file)
                json_run_file = open(run_f)
                json_run_data = json.load(json_run_file)

                times[index] = (dateutil.parser.parse(json_run_data['stop_time']) - dateutil.parser.parse(json_run_data['start_time'])).total_seconds()
                try:
                    if is_classification(model):
                        test_ccrs[index] = json_info_data['performance']['ccr']['test']
                        train_ccrs[index] = json_info_data['performance']['ccr']['train']
                        mss[index] = json_info_data['performance']['ms']
                        gms[index] = json_info_data['performance']['gm']
                    else:
                        test_mses[index] = json_info_data['performance']['mse']['test']
                        train_mses[index] = json_info_data['performance']['mse']['train']
                except KeyError:
                    print "WARNING: Experimento '%s' no completado." % info_f

            # Mean
            mean_results['ccr']['test'] = np.mean(test_ccrs)
            mean_results['ccr']['train'] = np.mean(train_ccrs)
            mean_results['mse']['test'] = np.mean(test_mses)
            mean_results['mse']['train'] = np.mean(train_mses)
            mean_results['ms'] = np.mean(mss)
            mean_results['gm'] = np.mean(gms)

            # Standard dev
            std_results['ccr']['test'] = np.std(test_ccrs)
            std_results['ccr']['train'] = np.std(train_ccrs)
            std_results['mse']['test'] = np.std(test_mses)
            std_results['mse']['train'] = np.std(train_mses)
            std_results['ms'] = np.std(mss)
            std_results['gm'] = np.std(gms)

            time = np.mean(times)

            with open(single_result_path + '/results.json', 'w') as fp:
                json.dump({
                    'mean': mean_results,
                    'std': std_results
                }, fp)





if __name__ == "__main__":
    save_results()

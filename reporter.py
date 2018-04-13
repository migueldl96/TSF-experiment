import datetime
import platform
import csv
import os
import numpy as np
from scipy.stats.mstats import gmean


class TSFReporter:
    def __init__(self):
        # Start date
        self.start_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Machine info
        self.platform = platform.platform()
        self.processor = platform.processor()
        self.system = platform.system()

        self.exec_time = None

        # Configuration info
        self.seed = None
        self.files = None
        self.test_ratio = None
        self.steps = None
        self.horizon = None
        self.param_grid = None
        self.model_config = None
        self.model_name = None

        # Result info
        self.best_conf = None
        self.real_train_serie = None
        self.predicted_train_serie = None
        self.real_test_serie = None
        self.predicted_test_serie = None

        # Regresssion performance
        self.train_mse = None
        self.test_mse = None

        # Classification performance
        self.train_ccr = None
        self.test_ccr = None
        self.min_sensitivity = None
        self.sensitivity_gm = None
        self.confusion_matrix = None

    def save(self):
        path = 'reports/' + self.start_date.replace(":", ".") + '/'
        deep_path = path + 'deep/'
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(deep_path):
            os.makedirs(deep_path)

        # Summary csv
        with open(path + '/summary.csv', 'w') as f:
            writer = csv.writer(f)
            rows = []

            # Header
            rows.append(["Date: ",  self.start_date])
            rows.append(["Platform: ", self.platform])
            rows.append(["Processor: ", self.processor])
            rows.append(["System: ", self.system])
            rows.append(["Execution time: ", "%.2fs" % self.exec_time])

            rows.append(["\n", ""])

            # Experiment configuration
            rows.append(["Seed: ", str(self.seed)])
            rows.append(["Endog serie file: ", self.files[0]])
            if len(self.files) > 1:
                rows.append(["Exog series files: ", self.files[1:]])
            rows.append(["Test ratio: ", self.test_ratio])
            rows.append(["Horizon: ", str(self.horizon)])
            rows.append(["Scoring: ", self.model_config['scoring']])
            rows.append(["Windows used:", ""])
            if self.steps['ar']:
                rows.append(["SimpleAR", "Parameters grid:"])
                for name, value in self.param_grid['ar'].iteritems():
                    rows.append(["", name + ": " + str(value)])
            if self.steps['dw']:
                rows.append(["DinamicWindow", "Parameters grid:"])
                for name, value in self.param_grid['dw'].iteritems():
                    rows.append(["", name + ": " + str(value)])
            if self.steps['cc']:
                rows.append(["ClassChange", "Parameters grid:"])
                for name, value in self.param_grid['cc'].iteritems():
                    rows.append(["", name + ": " + str(value)])
            rows.append(["Final estimator:", ""])
            if self.steps['model']:
                rows.append([self.model_name, "Parameters grid:"])
                for name, value in self.model_config['params'].iteritems():
                    rows.append(["", name + ": " + str(value)])

            rows.append(["\n", ""])

            # Results
            rows.append(["Windows best configuration:", ""])
            if self.steps['ar']:
                rows.append(["SimpleAR", "Best params:"])
                [rows.append(["", key + ": " + str(value)]) for key, value in self.best_conf.iteritems() if 'ar__' in key.lower()]
            if self.steps['dw']:
                rows.append(["DinamicWindow", "Best params:"])
                [rows.append(["", key + ": " + str(value)]) for key, value in self.best_conf.iteritems()
                 if 'dw__' in key.lower()]
            if self.steps['cc']:
                rows.append(["ClassChange", "Best params:"])
                [rows.append(["", key + ": " + str(value)]) for key, value in self.best_conf.iteritems()
                 if 'cc__' in key.lower()]
            if self.steps['model']:
                rows.append(["Estimator best configuration:", ""])
                rows.append([self.model_name, "Best params:"])
                [rows.append(["", key + ": " + str(value)]) for key, value in self.best_conf.iteritems()
                 if 'model__' in key.lower()]

            rows.append(["\n", ""])

            # Performance
            rows.append(["Experiment performance:", ""])
            if self.model_config['type'] == 'regression':
                rows.append(["Task type: ", "Regression"])
                rows.append(["Performance:", ""])
                rows.append(["Test MSE:", self.test_mse])
                rows.append(["Train MSE:", self.train_mse])
            elif self.model_config['type'] == 'classification':
                rows.append(["Task type: ", "Classification"])
                rows.append(["Performance:", ""])
                rows.append(["Test CCR:", self.test_ccr])
                rows.append(["Train CCR:", self.train_ccr])
                rows.append(["Minimun sensitivity: ", self.min_sensitivity])
                rows.append(["Sensitivity GM", self.sensitivity_gm])

            writer.writerows(rows)

        # Deep info
        # Real and predicted series on test/train
        train_data = np.asarray([self.real_train_serie, self.predicted_train_serie]).transpose()
        test_data = np.asarray([self.real_test_serie, self.predicted_test_serie]).transpose()
        format = '%f' if self.model_config['type'] == 'regression' else '%i'
        np.savetxt(deep_path + '/train.csv', train_data,
                   delimiter=',', fmt=format, header="Real,Predicted")
        np.savetxt(deep_path + '/test.csv', test_data,
                   delimiter=',', fmt=format, header="Real,Predicted")

        # Confusion matrix for classification tasks
        if self.model_config['type'] == 'classification':
            np.savetxt(deep_path + '/confusion_matrix.csv', self.confusion_matrix.astype(int), delimiter=',',
                       fmt='%i')

    def set_exec_time(self, time):
        self.exec_time = time

    def set_seed(self, seed):
        self.seed = seed

    def set_files(self, files):
        self.files = files

    def set_test_ratio(self, t_ratio):
        self.test_ratio = t_ratio

    def set_pipe_steps(self, steps):
        self.steps = steps

    def set_model_config(self, model_config, model_name):
        self.model_config = model_config
        self.model_name = model_name

    def set_horizon(self, horizon):
        self.horizon = horizon

    def set_param_grid(self, params):
        self.param_grid = params

    def set_best_configuration(self, best_config):
        self.best_conf = best_config

    def set_train_series(self, real, predicted):
        self.real_train_serie = real
        self.predicted_train_serie = predicted

    def set_test_series(self, real, predicted):
        self.real_test_serie = real
        self.predicted_test_serie = predicted

    def set_mse_performance(self, test, train):
        self.test_mse = test
        self.train_mse = train

    def set_ccr_performance(self, test, train):
        self.test_ccr = test
        self.train_ccr = train

    def set_sensitivity(self, tprs):
        self.min_sensitivity = min(tprs)
        self.sensitivity_gm = gmean(tprs)

    def set_confusion_matrix(self, cm):
        self.confusion_matrix = cm
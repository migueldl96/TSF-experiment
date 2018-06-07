import os
import numpy as np
import glob

class TSFReporter:
    def __init__(self, run_id, report_path):

        self.run_id = run_id
        self.report_path = report_path
        self.task_type = None
        # Result info
        self.real_train_serie = None
        self.predicted_train_serie = None
        self.real_test_serie = None
        self.predicted_test_serie = None

        # Classification performance
        self.confusion_matrix = None

    def save(self):
        deep_path = glob.glob(self.report_path + 'report_' + str(self.run_id) + '*')[0]
        deep_path += '/deep'
        if not os.path.exists(deep_path):
            os.makedirs(deep_path)

        # Deep info
        # Real and predicted series on test/train
        train_data = np.asarray([self.real_train_serie, self.predicted_train_serie]).transpose()
        test_data = np.asarray([self.real_test_serie, self.predicted_test_serie]).transpose()
        format = '%f' if self.task_type == 'regression' else '%i'
        np.savetxt(deep_path + '/train.csv', train_data,
                   delimiter=',', fmt=format, header="Real,Predicted")
        np.savetxt(deep_path + '/test.csv', test_data,
                   delimiter=',', fmt=format, header="Real,Predicted")

        # Confusion matrix for classification tasks
        if self.task_type == 'classification':
            np.savetxt(deep_path + '/confusion_matrix.csv', self.confusion_matrix.astype(int), delimiter=',',
                       fmt='%i')

    def set_train_series(self, real, predicted):
        self.real_train_serie = real
        self.predicted_train_serie = predicted

    def set_test_series(self, real, predicted):
        self.real_test_serie = real
        self.predicted_test_serie = predicted

    def set_confusion_matrix(self, cm):
        self.confusion_matrix = cm

    def set_task_type(self, type):
        self.task_type = type
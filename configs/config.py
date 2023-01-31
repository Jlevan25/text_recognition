import os
import time


class Config(object):
    def __init__(self, batch_size, lr, dataset_name, model_name, out_features, DATASET_DIR, ROOT_DIR=None,
                 debug=True, write_by_class_metrics: bool = True, weight_decay=None, momentum=0,
                 show_each=1, device='cpu', seed=None, overfit=False, shuffle=False):
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if ROOT_DIR is None else ROOT_DIR
        self.DATASET_DIR = DATASET_DIR
        self.debug = debug
        self.write_by_class_metrics = write_by_class_metrics
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.show_each = show_each
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.overfit = overfit
        self.seed = seed
        self.out_features = out_features
        self.shuffle = shuffle

        experiment_name = f'model_{self.model_name}_batch_size{self.batch_size}_lr_{self.lr}_{time.time()}'

        self.SAVE_PATH = os.path.join(self.ROOT_DIR, 'checkpoints', self.model_name,
                                      self.dataset_name, experiment_name)

        self.LOG_PATH = os.path.join(self.ROOT_DIR, 'logs', self.dataset_name, experiment_name)

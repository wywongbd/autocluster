from dataset import Dataset

#libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture, metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from algorithms import algorithms

np.random.seed(0)

class AutoCluster(object):
    def __init__(self):
        self._dataset = None
        self._algorithms_ls = None #usually this would be a list of algorithms

    def fit(self, dataset, algorithms_ls = [algorithms.KMeans, algorithms.DBSCAN]):
        # create dataset object
        self._dataset = Dataset(dataset)
        #config space object
        cs = build_config_space(algorithms_ls)
        #config space object

        #define scenario object to be passed
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": 50, #50 iterations
            "cs": cs,
            "deterministic": "true"})

        def f(cfg):
            cfg = {k : cfg[k] for k in cfg if cfg[k]}

            X = dataset
            algorithm = cluster.DBSCAN(cfg['eps'])
            algorithm.fit(X)

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            if len(set(y_pred)) == 1:
                return 0
            else:
                return -1 * metrics.silhouette_score(X, y_pred, metric='euclidean')

        smac_obj = SMAC(scenario=scenario, rng=np.random.RandomState(30), tae_runner=f)
        return smac_obj.optimize()

    def plot_clusters(self):
        pass

    def predict(self, X):
        return self.algorithm.predict(X) if self.algorithm else None

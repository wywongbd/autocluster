from dataset import Dataset
from algorithms import algorithms
from build_config_space import build_config_space, Mapper

#libraries
import numpy as np

from sklearn import cluster, datasets, mixture, metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

np.random.seed(0)

class AutoCluster(object):
    def __init__(self):
        self._dataset = None
        self._algorithms_ls = None #usually this would be a list of algorithms

    def fit(self, X, algorithms_ls = ["KMeans", "DBSCAN"]):
        # create dataset object
        self._dataset = Dataset(X)
        
        #config space object
        cs = build_config_space(algorithms_ls)

        #define scenario object to be passed
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": 50, #50 iterations
            "cs": cs,
            "deterministic": "true"
        })
        
        # this is the blackbox function to be optimized
        def f(cfg):
            # convert cfg into a dictionary
            cfg = {k : cfg[k] for k in cfg if cfg[k]}
            
            # remove keys with value == None
            cfg_subset = {k: v for k, v in cfg.items() if v is not None}
            
            # get the model chosen
            algorithm = Mapper.getClass(cfg_subset["algorithm_choice"])
            
            # pop "algorithm_choice" key from the dictionary
            cfg.pop("algorithm_choice", None)
            
            # standardize dataset
            scaled_data = self._dataset.standard_scaler.transform(X)
            
            # build model
            model = algorithm.model(**cfg)
            model.fit(scaled_data)

            if hasattr(model, 'labels_'):
                y_pred = model.labels_.astype(np.int)
            else:
                y_pred = model.predict(X)

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

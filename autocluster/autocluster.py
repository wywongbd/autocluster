from dataset import Dataset
from algorithms import algorithms
from build_config_space import build_config_space, Mapper
from utils.stringutils import StringUtils

import numpy as np
from sklearn import cluster, metrics

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class AutoCluster(object):
    def __init__(self):
        self._dataset = None
        self._algorithm = None

    def fit(self, X, 
            algorithms_ls=['KMeans','DBSCAN'], 
            n_evaluations=50, 
            seed=30):
        
        # create dataset object
        self._dataset = Dataset(X)
        
        # standardize dataset
        scaled_data = self._dataset.standard_scaler.transform(X)
        
        #config space object
        cs = build_config_space(algorithms_ls)
        
        print(cs)

        #define scenario object to be passed into SMAC
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": n_evaluations,
            "cs": cs,
            "deterministic": "true"
        })
        
        # helper function
        def fit_model(cfg):
            # convert cfg into a dictionary
            cfg = {k : cfg[k] for k in cfg if cfg[k]}
            
            # remove keys with value == None
            cfg_subset = {k: v for k, v in cfg.items() if v is not None}
            
            # get the model chosen
            algorithm = Mapper.getClass(cfg_subset["algorithm_choice"])
            
            # pop "algorithm_choice" key from the dictionary
            cfg_subset.pop("algorithm_choice", None)
            
            # decode the encoded parameters
            cfg_subset_decoded = {StringUtils.decode_parameter(k, algorithm.name): v for k, v in cfg_subset.items()}
                        
            # build model
            model = algorithm.model(**cfg_subset_decoded)
            model.fit(scaled_data)
            
            return model
        
        # this is the blackbox function to be optimized
        def evaluate_model(cfg):
            candidate_model = fit_model(cfg)

            if hasattr(candidate_model, 'labels_'):
                y_pred = candidate_model.labels_.astype(np.int)
            else:
                y_pred = candidate_model.predict(scaled_data)

            if len(set(y_pred)) == 1:
                return 0
            else:
                return -1 * metrics.silhouette_score(scaled_data, y_pred, metric='euclidean')
        
        # run SMAC to optimize 
        smac_obj = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=evaluate_model)
        optimal_config = smac_obj.optimize()
        
        # refit to get optimal model
        optimal_model = fit_model(optimal_config)
        self._algorithm = optimal_model
        
        print("Optimization is complete, the optimal configuration is {}".format(optimal_config))
        
        # return a pair
        return smac_obj, optimal_config

    def plot_clusters(self):
        pass

    def predict(self, X):
        return self._algorithm.predict(self._dataset.standard_scaler.transform(X)) if self._algorithm else None

from dataset import Dataset
from algorithms import algorithms
from build_config_space import build_config_space, Mapper
from utils.stringutils import StringUtils
from utils.logutils import LogUtils

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
from itertools import cycle, islice

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class AutoCluster(object):
    def __init__(self):
        self._dataset = None
        self._algorithm = None
        self._smac_obj = None

    def fit(self, X, 
            algorithms_ls=['KMeans','DBSCAN'], 
            n_evaluations=50, 
            seed=30,
            run_obj='quality',
            cutoff_time=60,
            shared_model=True, 
            evaluator=(lambda X, y_pred: float('inf') if len(set(y_pred)) == 1 \
                       else -1 * metrics.silhouette_score(X, y_pred, metric='euclidean'))  
           ):
        """
        --------------------------------
        Arguments:
        --------------------------------
        X: numpy array 
        algorithms_ls: list of clustering algorithms to explore
        n_evaluations: max # of evaluations done during optimization, higher values yield better results 
        run_obj: 'runtime' or 'quality', cutoff_time must be provided if 'runtime' chosen.
        cutoff_time: Maximum runtime, after which the target algorithm is cancelled. Required if run_obj is 'runtime'.
        shared_model: whether or not to use parallel SMAC 
        evaluator: a function for evaluating clustering result, must have the arguments X and y_pred
        """
        
        # create dataset object
        self._dataset = Dataset(X)
        
        # standardize dataset
        scaled_data = self._dataset.standard_scaler.transform(X)
        
        #config space object
        cs = build_config_space(algorithms_ls)
        
        print(cs)

        #define scenario object to be passed into SMAC
        scenario = Scenario({
            "run_obj": run_obj,
            "runcount-limit": n_evaluations,
            "cutoff_time": cutoff_time,
            "cs": cs,
            "deterministic": "true",
            "input_psmac_dirs": LogUtils.create_new_directory('psmac'),
            "output_dir": LogUtils.create_new_directory('smac'),
            "shared_model": shared_model
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

            return evaluator(X=scaled_data, y_pred=y_pred)
        
        # run SMAC to optimize 
        self._smac_obj = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=evaluate_model)
        optimal_config = self._smac_obj.optimize()
        
        # refit to get optimal model
        self._algorithm = fit_model(optimal_config)
        
        print("Optimization is complete.")
        print("Took {} seconds, the optimal configuration is \n{}".format(self._smac_obj.stats.ta_time_used, 
                                                                          optimal_config))
        
        # return a pair
        return self._smac_obj, optimal_config

    def predict(self, X):
        if self._algorithm is None:
            return None
        
        scaled_X = self._dataset.standard_scaler.transform(X)
        y_pred = None
        
        try:
            y_pred = self._algorithm.predict(scaled_X)
        except:
            y_pred = self._algorithm.fit_predict(scaled_X) 
            
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=5, color=colors[y_pred])
        plt.show()
        return y_pred
    
    def plot_convergence(self):
        if self._smac_obj is None:
            return
        
        history = self._smac_obj.runhistory.data
        cost_ls = [v.cost for k, v in history.items()]
        min_cost_ls = list(np.minimum.accumulate(cost_ls))
        
        # plotting
        plt.figure(figsize=(10,10))
        plt.plot(min_cost_ls, linestyle='-', marker='o', color='b')
        plt.xlabel('n_evaluations', color='white', fontsize=15)
        plt.ylabel('performance of best configuration', color='white', fontsize=15)
        plt.show()
        
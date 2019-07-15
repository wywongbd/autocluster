from dataset import Dataset
from algorithms import algorithms
from build_config_space import build_config_space, Mapper
from utils.stringutils import StringUtils
from utils.logutils import LogUtils

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, metrics, manifold
from itertools import cycle, islice

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class AutoCluster(object):
    def __init__(self, logger=None):
        self._dataset = None
        self._clustering_model = None
        self._dim_reduction_model = None
        self._smac_obj = None
        self._logger = logger

    def fit(self, X, 
            cluster_alg_ls=['KMeans','DBSCAN'],
            dim_reduction_alg_ls=[],
            n_evaluations=50, 
            seed=30,
            run_obj='quality',
            cutoff_time=60,
            shared_model=True,
            n_parallel_runs=3,
            evaluator=(lambda X, y_pred: float('inf') if len(set(y_pred)) == 1 \
                       else -1 * metrics.silhouette_score(X, y_pred, metric='euclidean'))  
           ):
        """
        --------------------------------
        Arguments:
        --------------------------------
        X: numpy array 
        cluster_alg_ls: list of clustering algorithms to explore
        dim_reduction_alg_ls: list of dimension algorithms to explore
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
        cs = build_config_space(cluster_alg_ls, dim_reduction_alg_ls)
        
        # make sure n_evaluations is valid
        dim_reduction_min_size = 1 if len(dim_reduction_alg_ls) == 0 \
                                else min([Mapper.getClass(alg).n_possible_cfgs 
                                          for alg in dim_reduction_alg_ls])
        clustering_min_size = min([Mapper.getClass(alg).n_possible_cfgs for alg in cluster_alg_ls])
        n_evaluations = min(n_evaluations, clustering_min_size * dim_reduction_min_size)
        
        self._log(cs)
        self._log('Truncated n_evaluations: {}'.format(n_evaluations))

        #define scenario object to be passed into SMAC
        scenario_params = {
            "run_obj": run_obj,
            "runcount-limit": n_evaluations,
            "cutoff_time": cutoff_time,
            "cs": cs,
            "deterministic": "true",
            "input_psmac_dirs": [LogUtils.create_new_directory('psmac') for i in range(n_parallel_runs)] 
                                if shared_model else None,
            "output_dir": LogUtils.create_new_directory('smac'),
            "shared_model": shared_model,
            "abort_on_first_run_crash": False,
        }
        scenario = Scenario(scenario_params)
        
        self._log('{}'.format(scenario_params))
        
        # helper function
        def fit_model(cfg):
            compressed_data = scaled_data
            
            # convert cfg into a dictionary
            cfg = {k : cfg[k] for k in cfg if cfg[k]}
            
            # remove keys with value == None
            cfg = {k: v for k, v in cfg.items() if v is not None}
            
            self._log("Fitting configuration: {}".format(cfg))
            
            # get the dimension reduction method chosen
            dim_reduction_alg = Mapper.getClass(cfg.get("dim_reduction_choice", None))
            dim_reduction_model = None
            
            # fit dimension reduction model
            if dim_reduction_alg:
                cfg_dim_reduction = {StringUtils.decode_parameter(k, dim_reduction_alg.name): v
                                     for k, v in cfg.items() if StringUtils.decode_parameter(k, dim_reduction_alg.name) is not None}
                
                # compress the data using chosen configurations
                dim_reduction_model = dim_reduction_alg.model(**cfg_dim_reduction)
                compressed_data = dim_reduction_model.fit_transform(scaled_data)
            
            # get the model chosen
            clustering_alg = Mapper.getClass(cfg["clustering_choice"])
               
            # decode the encoded parameters
            cfg_clustering = {StringUtils.decode_parameter(k, clustering_alg.name): v 
                              for k, v in cfg.items() if StringUtils.decode_parameter(k, clustering_alg.name) is not None}
                        
            # build model
            clustering_model = clustering_alg.model(**cfg_clustering)
            clustering_model.fit(compressed_data)
            
            return clustering_model, dim_reduction_model, compressed_data
        
        # this is the blackbox function to be optimized
        def evaluate_model(cfg):
            candidate_model, _, compressed_data = fit_model(cfg)

            if hasattr(candidate_model, 'labels_'):
                y_pred = candidate_model.labels_.astype(np.int)
            else:
                y_pred = candidate_model.predict(compressed_data)
    
            return evaluator(X=compressed_data, y_pred=y_pred)
        
        # run SMAC to optimize 
        self._smac_obj = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=evaluate_model)
        optimal_config = self._smac_obj.optimize()
        
        # refit to get optimal model
        self._clustering_model, self._dim_reduction_model, _ = fit_model(optimal_config)
        
        self._log("Optimization is complete.")
        self._log("Took {} seconds.".format(round(self._smac_obj.stats.get_used_wallclock_time(), 2)))
        self._log("The optimal configuration is \n{}".format(optimal_config))
        
        # return a pair
        return self._smac_obj, optimal_config

    def predict(self, X , plot=True):
        if self._clustering_model is None:
            return None
        
        scaled_X = self._dataset.standard_scaler.transform(X)
        
        if self._dim_reduction_model:
            try:
                compressed_X = self._dim_reduction_model.transform(scaled_X)
            except:
                compressed_X = self._dim_reduction_model.fit_transform(scaled_X)
        else:
            compressed_X = scaled_X
        
        y_pred = None
        
        try:
            y_pred = self._clustering_model.predict(compressed_X)
        except:
            y_pred = self._clustering_model.fit_predict(compressed_X) 
        
        if plot:
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', 'magenta',
                                                 '#f781bf', '#a65628', '#984ea3', 'black',
                                                 '#999999', '#e41a1c', '#dede00', 'cyan']),
                                                  int(max(y_pred) + 1))))
            # check if dimension reduction is needed
            if compressed_X.shape[1] > 2:
                self._log('performing TSNE')
                compressed_X = manifold.TSNE(n_components=2).fit_transform(compressed_X) 
                
            plt.figure(figsize=(10,10))
            plt.scatter(compressed_X[:, 0], compressed_X[:, 1], s=7, color=colors[y_pred])
            plt.tick_params(axis='x', colors='white')
            plt.tick_params(axis='y', colors='white')
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
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        plt.show()
        
    def _log(self, string):
        if self._logger:
            self._logger.info(string)
        else:
            print(string)
        
from algorithms import algorithms
from warmstarter import KDTreeWarmstarter
from preprocess_data import PreprocessedDataset
from utils.stringutils import StringUtils
from utils.logutils import LogUtils
from utils.constants import Constants 
from utils.metafeatures import calculate_metafeatures, MetafeatureMapper
from build_config_space import build_config_space, build_config_obj, Mapper

from itertools import cycle, islice
from sklearn import cluster, metrics, manifold, ensemble, model_selection, preprocessing

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.optimizer import smbo, pSMAC

import os
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AutoCluster(object):
    def __init__(self, logger=None):
        self._dataset = None
        self._clustering_model = None
        self._dim_reduction_model = None
        self._scaler = None
        self._preprocess_dict = None
        self._smac_obj = None
        self._logger = logger
        self._log_path = None
        
        if self._logger:
            self._log_path = logging.getLoggerClass().root.handlers[0].baseFilename
    
    def fit(self, df, 
            cluster_alg_ls=['KMeans','DBSCAN'],
            dim_reduction_alg_ls=[],
            n_evaluations=30,
            run_obj='quality',
            seed=27,
            cutoff_time=50,
            evaluator=(lambda X, y_pred: float('inf') if len(set(y_pred)) == 1 \
                   else -1 * metrics.silhouette_score(X, y_pred, metric='euclidean')),
            preprocess_dict={},
            warmstart=False,
            warmstart_datasets_dir='silhouette',
            general_metafeatures=[],
            numeric_metafeatures=[],
            categorical_metafeatures=[]
           ):
        """
        ---------------------------------------------------------------------------
        Arguments
        ---------------------------------------------------------------------------
        df: a DataFrame
        preprocess_dict: should be a dictionary with keys 'numeric_cols', 'ordinal_cols', 'categorical_cols' and 'y_col'
        cluster_alg_ls: list of clustering algorithms to explore
        dim_reduction_alg_ls: list of dimension algorithms to explore
        n_evaluations: max # of evaluations done during optimization, higher values yield better results 
        run_obj: 'runtime' or 'quality', cutoff_time must be provided if 'runtime' chosen.
        cutoff_time: Maximum runtime, after which the target algorithm is cancelled. Required if run_obj is 'runtime'.
        shared_model: whether or not to use parallel SMAC 
        evaluator: a function for evaluating clustering result, must have the arguments X and y_pred   
        """
        #############################################################
        # Data preprocessing                                        #
        #############################################################
        # rename, save preprocess_dict for later use
        raw_data = df
        self._preprocess_dict = preprocess_dict
        
        # encode categorical and ordinal columns
        preprocess_dict['df'] = raw_data
        raw_data_np = PreprocessedDataset(**preprocess_dict).X
        
        # perform outlier detection
        predicted_labels = ensemble.IsolationForest(n_estimators=100, 
                                                    warm_start=True,
                                                    behaviour='new',
                                                    contamination='auto').fit_predict(raw_data_np)
        idx_np = np.where(predicted_labels == 1)
        
        # remove outliers
        raw_data_cleaned = raw_data.iloc[idx_np].reset_index(drop=True)
        self._log("{}/{} datapoints remaining after outlier removal".format(len(raw_data_cleaned), 
                                                                            len(raw_data_np)))
        
        # encode cleaned datasest
        preprocess_dict['df'] = raw_data_cleaned
        processed_data_np = PreprocessedDataset(**preprocess_dict).X
        
        #############################################################
        # Warmstarting (Optional)                                   #
        #############################################################
        
        # construct desired configuration space
        cs = build_config_space(cluster_alg_ls, dim_reduction_alg_ls)
        self._log(cs)    
        
        # calculate metafeatures
        metafeatures_np = None
        metafeatures_ls = general_metafeatures + numeric_metafeatures + categorical_metafeatures
        if len(metafeatures_ls) > 0:
            metafeatures_np = calculate_metafeatures(raw_data_cleaned, preprocess_dict, 
                                                     metafeatures_ls)
        
        # perform warmstart, if needed
        initial_cfgs_ls = []
        if warmstart and len(metafeatures_ls) > 0:
            # create and train warmstarter 
            warmstarter = KDTreeWarmstarter(metafeatures_ls)
            warmstarter.fit()
            
            # query for suitable configurations
            initial_configurations = warmstarter.query(metafeatures_np, 3, 20, 
                                                       datasets_dir=warmstart_datasets_dir)
            
            # construct configuration objects
            for cfg in initial_configurations:
                try:
                    initial_cfgs_ls.append(build_config_obj(cs, cfg[0]))
                except:
                    pass
                
        # if too little configurations available, just ignore
        initial_cfgs_ls = None if len(initial_cfgs_ls) < 2 else initial_cfgs_ls
            
        #############################################################
        # Bayesian optimization (SMAC)                              #
        #############################################################
        # make sure n_evaluations is valid
        dim_reduction_min_size = 1 if len(dim_reduction_alg_ls) == 0 \
                                else min([Mapper.getClass(alg).n_possible_cfgs 
                                          for alg in dim_reduction_alg_ls])
        clustering_min_size = min([Mapper.getClass(alg).n_possible_cfgs for alg in cluster_alg_ls])
        n_evaluations = min(n_evaluations, clustering_min_size * dim_reduction_min_size)
        initial_cfgs_ls = initial_cfgs_ls[0 : n_evaluations] if initial_cfgs_ls is not None else None
        self._log('Truncated n_evaluations: {}'.format(n_evaluations))
        
        # define scenario object to be passed into SMAC
        scenario_params = {
            "run_obj": run_obj,
            "runcount-limit": n_evaluations,
            "cutoff_time": cutoff_time,
            "cs": cs,
            "deterministic": "true",
            "output_dir": LogUtils.create_new_directory('{}/smac'.format(self.log_dir)),
            "abort_on_first_run_crash": False,
        }
        scenario = Scenario(scenario_params)    
        self._log('{}'.format(scenario_params))
        
        # functions required for SMAC optimization
        def fit_models(cfg, data):
            ################################################
            # Get configurations                           #
            ################################################
            # convert cfg into a dictionary
            cfg = {k : cfg[k] for k in cfg if cfg[k]}
            
            # remove keys with value == None
            cfg = {k: v for k, v in cfg.items() if v is not None}
            
            # logging
            self._log("Fitting configuration: \n{}".format(cfg))
            
            ################################################
            # Preprocessing                                #
            ################################################
            # fit standard scaler
            scaler = preprocessing.StandardScaler()
            scaler.fit(data)
            
            # standardize data
            scaled_data = scaler.transform(data)
            
            ################################################
            # Dimensionality reduction                     #
            ################################################
            # get the dimension reduction method chosen
            dim_reduction_alg = Mapper.getClass(cfg.get("dim_reduction_choice", None))
            dim_reduction_model = None
            
            # fit dimension reduction model
            compressed_data = scaled_data
            if dim_reduction_alg:
                cfg_dim_reduction = {StringUtils.decode_parameter(k, dim_reduction_alg.name): v
                                     for k, v in cfg.items() if StringUtils.decode_parameter(k, dim_reduction_alg.name) is not None}
                
                # compress the data using chosen configurations
                dim_reduction_model = dim_reduction_alg.model(**cfg_dim_reduction)
                compressed_data = dim_reduction_model.fit_transform(scaled_data)
            
            ################################################
            # Clustering                                   #
            ################################################
            # get the model chosen
            clustering_alg = Mapper.getClass(cfg["clustering_choice"])
               
            # decode the encoded parameters
            cfg_clustering = {StringUtils.decode_parameter(k, clustering_alg.name): v 
                              for k, v in cfg.items() if StringUtils.decode_parameter(k, clustering_alg.name) is not None}
                        
            # train clustering model
            clustering_model = clustering_alg.model(**cfg_clustering)
            clustering_model.fit(compressed_data)
            
            return scaler, dim_reduction_model, clustering_model, 
        
        def evaluate_model(cfg):
            # split data into train and test
            train_data, valid_data = model_selection.train_test_split(processed_data_np,
                                                                      test_size=0.4, 
                                                                      random_state=seed,
                                                                      shuffle=True)
            
            # fit clustering and dimension reduction models on training data
            scaler, dim_reduction_model, clustering_model = fit_models(cfg, train_data)
            
            # test on validation data
            scaled_valid_data = scaler.transform(valid_data)
            compressed_valid_data = scaled_valid_data
            if dim_reduction_model:
                try:
                    compressed_valid_data = dim_reduction_model.transform(scaled_valid_data)
                except:
                    compressed_valid_data = dim_reduction_model.fit_transform(scaled_valid_data)
                    
            # predict on validation data
            if hasattr(clustering_model, 'fit_predict'):
                y_pred = clustering_model.fit_predict(compressed_valid_data)
            else:
                y_pred = clustering_model.predict(compressed_valid_data)
            
            # evaluate using provided evaluator
            score = evaluator(X=compressed_valid_data, y_pred=y_pred)
            self._log("Score obtained by this configuration: {}".format(score))
            
            return score
    
        # run SMAC to optimize
        smac_params = {
            "scenario": scenario,
            "rng": np.random.RandomState(seed),
            "tae_runner": evaluate_model,
            "initial_configurations": initial_cfgs_ls,
        }
        self._smac_obj = SMAC(**smac_params)
        optimal_config = self._smac_obj.optimize()
        
        # refit to get optimal model
        self._scaler, self._dim_reduction_model, self._clustering_model = fit_models(optimal_config, processed_data_np)
        self._log("Optimization is complete.")
        self._log("Took {} seconds.".format(round(self._smac_obj.stats.get_used_wallclock_time(), 2)))
        self._log("The optimal configuration is \n{}".format(optimal_config))
        
        # return a dictionary
        result = {
            "cluster_alg_ls": cluster_alg_ls,
            "dim_reduction_alg_ls": dim_reduction_alg_ls,
            "smac_obj": self._smac_obj,
            "optimal_cfg": optimal_config,
            "metafeatures": metafeatures_np,
            "metafeatures_used": metafeatures_ls,
            "clustering_model": self._clustering_model,
            "dim_reduction_model": self._dim_reduction_model,
            "scaler": self._scaler
        }
        return result
    

    def predict(self, df, plot=True):
        if (self._clustering_model is None) or (self._preprocess_dict is None):
            return None
        
        # encode data using preprocess_dict
        data = df
        self._preprocess_dict['df'] = data
        data_np = PreprocessedDataset(**self._preprocess_dict).X
        
        # scale data
        scaled_data = self._scaler.transform(data_np)

        # dimensionality reduction
        compressed_data = scaled_data
        if self._dim_reduction_model:
            try:
                compressed_data = self._dim_reduction_model.transform(scaled_data)
            except:
                compressed_data = self._dim_reduction_model.fit_transform(scaled_data)
        
        # prediction
        y_pred = None
        try:
            y_pred = self._clustering_model.predict(compressed_data)
        except:
            y_pred = self._clustering_model.fit_predict(compressed_data) 
        
        if plot:
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', 'magenta',
                                                 '#f781bf', '#a65628', '#984ea3', 'black',
                                                 '#999999', '#e41a1c', '#dede00', 'cyan']),
                                                  int(max(y_pred) + 1))))
            # check if dimension reduction is needed
            if compressed_data.shape[1] > 2:
                self._log('performing TSNE')
                compressed_data = manifold.TSNE(n_components=2).fit_transform(compressed_data) 
                
            plt.figure(figsize=(10,10))
            plt.scatter(compressed_data[:, 0], compressed_data[:, 1], s=7, color=colors[y_pred])
            plt.tick_params(axis='x', colors='white')
            plt.tick_params(axis='y', colors='white')
            plt.show()
            
        return y_pred
    
    def get_trajectory(self):
        if self._smac_obj is None:
            return None
        return [(vars(t.incumbent)['_values'], t.train_perf) for t in self._smac_obj.get_trajectory()] 
    
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
    
    @property
    def log_dir(self):
        return '/{}'.format(self._log_path.split(os.sep)[-2]) if self._logger else ''

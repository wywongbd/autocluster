from dataset import Dataset
from preprocess_data import PreprocessedDataset
from algorithms import algorithms
from build_config_space import build_config_space, Mapper, build_config_obj
from utils.stringutils import StringUtils
from utils.logutils import LogUtils
from utils.constants import Constants 
from utils.metafeatures import calculate_metafeatures, MetafeatureMapper
from warmstarter import KDTreeWarmstarter

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
        self._smac_obj = None
        self._logger = logger
        self._log_path = None
        
        if self._logger:
            self._log_path = logging.getLoggerClass().root.handlers[0].baseFilename

    def fit(self, X, 
            cluster_alg_ls=['KMeans','DBSCAN'],
            dim_reduction_alg_ls=[],
            n_evaluations=50, 
            seed=30,
            run_obj='quality',
            cutoff_time=60,
            shared_model=True,
            n_parallel_runs=3,
            initial_configurations=None,
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

        # define scenario object to be passed into SMAC
        scenario_params = {
            "run_obj": run_obj,
            "runcount-limit": n_evaluations,
            "cutoff_time": cutoff_time,
            "cs": cs,
            "deterministic": "true",
            "input_psmac_dirs": [
                LogUtils.create_new_directory('{}/psmac'.format(self.log_dir)) for i in range(n_parallel_runs)
            ] if shared_model else None,
            "output_dir": LogUtils.create_new_directory('{}/smac'.format(self.log_dir)),
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
            
            # logging
            self._log("Fitting configuration: \n{}".format(cfg))
            
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
                
            score = evaluator(X=compressed_data, y_pred=y_pred)
            self._log("Score obtained by this configuration: {}".format(score))
            return score
        
        # build configuration objects
        if initial_configurations:
            ls = []
            for cfg in initial_configurations:
                try:
                    obj = build_config_obj(cs, cfg[0])
                    ls.append(obj)
                except:
                    pass
            initial_configurations = ls[0 : n_evaluations]
        
        # run SMAC to optimize
        smac_params = {
            "scenario": scenario,
            "rng": np.random.RandomState(seed),
            "tae_runner": evaluate_model,
            "initial_configurations": initial_configurations,
        }
        self._smac_obj = SMAC(**smac_params)
        optimal_config = self._smac_obj.optimize()
        
        # refit to get optimal model
        self._clustering_model, self._dim_reduction_model, _ = fit_model(optimal_config)
        
        self._log("Optimization is complete.")
        self._log("Took {} seconds.".format(round(self._smac_obj.stats.get_used_wallclock_time(), 2)))
        self._log("The optimal configuration is \n{}".format(optimal_config))
        
        # return a pair
        return self._smac_obj, optimal_config
    
    def fit_raw_data(self, df, 
                     n_evaluations=30,
                     seed=27,
                     cutoff_time=50,
                     numerical_cols=[],
                     categorical_cols=[],
                     warmstart=True,
                     general_metafeatures=['all'],
                     numeric_metafeatures=[],
                     categorical_metafeatures=[]
                    ):
        # load warmstarter
        warmstarter = KDTreeWarmstarter(Constants.default_general_metafeatures)
        warmstarter.fit()
        
        # load constants
        default_general_metafeatures_set = set(Constants.default_general_metafeatures)
        default_numeric_metafeatures_set = set(Constants.default_numeric_metafeatures)
        
        # compute metafeatures
        feature = {} 
        m_ls = [m for m in warmstarter.metafeatures if m != 'dataset']
        
        for m in m_ls:
            if m in default_general_metafeatures_set:
                feature[m] = getattr(Metafeatures, m)(df.to_numpy())
            elif m in default_numeric_metafeatures_set:
                if len(numerical_cols) > 0:
                    feature[m] = getattr(Metafeatures, m)(df[numerical_cols].to_numpy())
                else:
                    feature = None
                    break
        
        # get initial configurations
        initial_configurations = None
        if feature is not None:
            feature = pd.DataFrame([feature])[m_ls].to_numpy()
            initial_configurations = warmstarter.query(feature, 3, 20)
        
        # fit on data
        fit_config = {
            "X": df.to_numpy(), 
            "cluster_alg_ls": [
                'KMeans', 'GaussianMixture',
                'MiniBatchKMeans', 'AgglomerativeClustering'
            ], 
            "dim_reduction_alg_ls": [
                'TSNE', 'PCA', 'IncrementalPCA', 
                'FastICA', 'TruncatedSVD'
            ],
            "n_evaluations": n_evaluations,
            "seed": seed, 
            "run_obj": 'quality', 
            "cutoff_time": cutoff_time, 
            "shared_model": False,
            "n_parallel_runs": 3,
            "evaluator": lambda X, y_pred: 
                            float('inf') if len(set(y_pred)) == 1 \
                            else -1 * metrics.silhouette_score(X, y_pred),
            "initial_configurations": initial_configurations
        }
        
        return self.fit(**fit_config)
    
    def _fit(self, df, 
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
             general_metafeatures=[],
             numeric_metafeatures=[],
             categorical_metafeatures=[]
            ):
        """
        preprocess_dict: should be a dictionary with keys numeric_cols, ordinal_cols, categorical_cols and ignore_cols
        """
        #############################################################
        # Data preprocessing                                        #
        #############################################################
        # rename 
        raw_data = df
        
        # encode categorical and ordinal columns
        preprocess_dict['df'] = raw_data
        raw_data_np = PreprocessedDataset(**preprocess_dict).X
        
        # perform outlier detection
        predicted_labels = ensemble.IsolationForest(n_estimators=20, 
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
        
        # perform warmstart, if needed
        initial_cfgs_ls = None
        if warmstart:
            # create and train warmstarter 
            warmstarter = KDTreeWarmstarter(general_metafeatures + numeric_metafeatures + \
                                            categorical_metafeatures)
            warmstarter.fit()
            
            # calculate metafeatures
            metafeatures = calculate_metafeatures(raw_data_cleaned,
                                                  preprocess_dict,
                                                  general_metafeatures + numeric_metafeatures + 
                                                  categorical_metafeatures)
            print(metafeatures)
            
            # query for suitable configurations
            initial_configurations = warmstarter.query(metafeatures, 3, 20)
            
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
        
        # return a pair
        return self._smac_obj, optimal_config
    

    def predict(self, X, plot=True):
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

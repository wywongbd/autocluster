import glob
import os
import sys
import argparse
import pathlib
import logging
import json
import random
import traceback
import numpy as np
import pandas as pd

# change directory
sys.path.append("../")

import autocluster
from autocluster import AutoCluster, PreprocessedDataset, get_evaluator, LogHelper, LogUtils, MetafeatureMapper, calculate_metafeatures

from sklearn import datasets
from collections import Counter
from sklearn.metrics.cluster import v_measure_score
from sklearn.model_selection import train_test_split
from datetime import datetime

##################################################################################################
# Define parameters for script                                                                   #
##################################################################################################
parser = argparse.ArgumentParser()

# directories
parser.add_argument("--benchmark_metafeatures_table_path", 
                    default='metaknowledge/benchmark_silhouette_metafeatures_table.csv', 
                    type=str, help="Path to benchmark metafeatures table.")

# logging 
parser.add_argument("--log_dir_prefix", type=str, default='benchmark_experiment', 
                    help='Prefix of directory')

# optimization
parser.add_argument("--test_size", default=0.1666, type=float,
                    help="Portion of benchmark datasets used for testing.")
parser.add_argument('--optimizer', choices=['smac', 'random'], default='smac', 
                    help='Choice of optimizer, smac for bayesian optimization, random for random sampling optimization.')
parser.add_argument("--n_folds", default=3, type=int,
                    help="Number of folds used in k-fold cross validation during evaluation step of optimization.")
parser.add_argument("--random_seed", default=27, type=int,
                    help="Random seed used in optimization.")
parser.add_argument("--n_evaluations", default=10, type=int, 
                    help="Number of evaluations used in optimization.")
parser.add_argument("--cutoff_time", default=100, type=int, 
                    help="Configuration will be terminated if it takes > cutoff_time to run.")
parser.add_argument("--warmstart", default=1, type=int, 
                    help="Flag to indicate whether to use warmstart, 0 for no, 1 for yes.")
parser.add_argument("--warmstart_n_neighbors", default=3, type=int, 
                    help="Number of similar datasets to use for retrieving initial configurations of warmstarter.")
parser.add_argument("--warmstart_top_n", default=10, type=int, 
                    help="Number of initial configurations to retrieve from each similar dataset.")
parser.add_argument("--general_metafeatures", default=MetafeatureMapper.getGeneralMetafeatures(), nargs='+', type=str,
                    help="List of general metafeatures to use.")
parser.add_argument("--numeric_metafeatures", default=MetafeatureMapper.getNumericMetafeatures(), nargs='+', type=str, 
                    help="List of numeric meteafeatures to use.")

config = parser.parse_args()

##################################################################################################
# Helper functions                                                                               #
##################################################################################################

def read_json_file(filename):
   with open(filename) as f_in:
       return(json.load(f_in))
    
##################################################################################################
# Main function                                                                                  #
##################################################################################################

def main():
    # Create output directory
    output_dir = LogUtils.create_new_directory(prefix=config.log_dir_prefix)    

    # Setup logger
    LogHelper.setup(log_path='{}/meta.log'.format(output_dir), log_level=logging.INFO)
    _logger = logging.getLogger(__name__)
    _logger_path = logging.getLoggerClass().root.handlers[0].baseFilename
    _logger.info("Log file location: {}".format(_logger_path))
    
    # log all arguments passed into this script
    _logger.info("Script arguments: {}".format(vars(config)))
    
    # set random seed program-wide
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # read all available benchmark datasets
    metafeatures_table = pd.read_csv(config.benchmark_metafeatures_table_path, 
                                     sep=',', header='infer')
    _logger.info('Found {} benchmark datasets: {}'.format(len(metafeatures_table), 
                                                          list(metafeatures_table['dataset'])))
    
    # split benchmark data into train and test
    train_idx, test_idx = train_test_split([i for i in range(len(metafeatures_table))], 
                                           test_size=config.test_size, random_state=config.random_seed)
    metafeatures_table_train, metafeatures_table_test = metafeatures_table.iloc[train_idx, :], metafeatures_table.iloc[test_idx, :]
    metafeatures_table_train, metafeatures_table_test = metafeatures_table_train.reset_index(drop=True), \
                                                        metafeatures_table_test.reset_index(drop=True)
    
    # save the train metafeatures table
    metafeatures_table_name_no_ext, _ = os.path.splitext(config.benchmark_metafeatures_table_path)
    metafeatures_table_train.to_csv('{}_trimmed.csv'.format(metafeatures_table_name_no_ext), 
                                    encoding='utf-8', index=False)
    
    # get names of test datasets
    test_datasets = list(metafeatures_table_test['dataset'])
    _logger.info('There are {} benchmark datasets chosen as test datasets: {}'.format(len(test_datasets), 
                                                                                      test_datasets))
    
    for i, dataset_name in enumerate(test_datasets, 1):
        # logging
        _logger.info("ITERATION {} of {}".format(i, len(test_datasets)))
        _logger.info("Running on dataset: {}".format(dataset_name))
        dataset_basename, _ = os.path.splitext(dataset_name)
        dataset = pd.read_csv('../data/benchmark_data/{}.csv'.format(dataset_basename), 
                              header='infer', sep=',')
        
        # for recording
        records = {}
        records["dataset"] = dataset_name

        # prepare dictionary for preprocessing
        preprocess_dict = read_json_file('../data/benchmark_data/{}.json'.format(dataset_basename))
        preprocess_dict = {k: v for k, v in preprocess_dict.items() 
                           if k in ["numeric_cols", "categorical_cols", "ordinal_cols", "y_col", "ignore_cols"]}

        # for safety reasons
        preprocess_dict["numeric_cols"] = preprocess_dict.get("numeric_cols", [])
        preprocess_dict["categorical_cols"] = preprocess_dict.get("categorical_cols", [])
        preprocess_dict["ordinal_cols"] = preprocess_dict.get("ordinal_cols", {})
        preprocess_dict["y_col"] = preprocess_dict.get("y_col", None)
        preprocess_dict["ignore_cols"] = preprocess_dict.get("ignore_cols", [])

        fit_params = {
            "df": dataset, 
            "cluster_alg_ls": [
                'KMeans', 'GaussianMixture', 'Birch', 
                'MiniBatchKMeans', 'AgglomerativeClustering', 'OPTICS', 
                'SpectralClustering', 'DBSCAN', 'AffinityPropagation', 'MeanShift'
            ], 
            "dim_reduction_alg_ls": [
                'TSNE', 'PCA', 'IncrementalPCA', 
                'KernelPCA', 'FastICA', 'TruncatedSVD',
                'NullModel'
            ],
            "optimizer": config.optimizer,
            "n_evaluations": config.n_evaluations,
            "run_obj": 'quality',
            "seed": config.random_seed,
            "cutoff_time": config.cutoff_time,
            "preprocess_dict": preprocess_dict,
            "evaluator": get_evaluator(evaluator_ls = ['silhouetteScore'], 
                                       weights = [], clustering_num = None, 
                                       min_proportion = .01),
            "n_folds": config.n_folds,
            "warmstart": (config.warmstart != 0),
            "warmstart_datasets_dir": 'benchmark_silhouette_v0',
            "warmstart_metafeatures_table_path": '{}_trimmed.csv'.format(metafeatures_table_name_no_ext),
            "warmstart_n_neighbors": config.warmstart_n_neighbors,
            "warmstart_top_n": config.warmstart_top_n,
            "general_metafeatures": config.general_metafeatures,
            "numeric_metafeatures": config.numeric_metafeatures,
            "categorical_metafeatures": [],
        }
        
        # fitting
        cluster = AutoCluster(logger=_logger)
        result = cluster.fit(**fit_params)
        predictions = cluster.predict(dataset, plot=False, 
                                      save_plot=True, file_path='{}/{}.png'.format(output_dir, dataset_name))
        _logger.info("Statistics of predictions: {}".format(Counter(predictions)))
        
        # save result
        if config.optimizer != 'random':
            records["trajectory"] = cluster.get_trajectory()
        metafeatures_ls = list(result["metafeatures"][0])
        records.update(dict(zip(result["metafeatures_used"], metafeatures_ls))) 
        
        # log results
        _logger.info("Done optimizing on {}.".format(dataset_name))
        _logger.info("Record on ITERATION {}: \n{}".format(i, records))
        _logger.info("Done with ITERATION {}.".format(i))
        
        
if __name__ == '__main__':
    main()

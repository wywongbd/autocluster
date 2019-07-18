import glob
import os
import sys
import argparse
import pathlib
import logging
import json
import random
import numpy as np
import pandas as pd

from autocluster import AutoCluster
from preprocess_data import PreprocessedDataset
from log_helper.log_helper import LogHelper
from utils.logutils import LogUtils
from utils.metafeatures import Metafeatures

from sklearn import datasets
from datetime import datetime
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

##################################################################################################
# Define parameters for script                                                                   #
##################################################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--raw_data_path", type=str, default="../data/raw_data/",
                    help="Directory of raw datasets.")
parser.add_argument("--processed_data_path", type=str, default='../data/processed_data/', 
                    help="Directory of processed datasets.")
parser.add_argument("--log_dir_prefix", type=str, default='meta_learning', 
                    help='Prefix of directory')

parser.add_argument("--n_parallel_runs", default=3, type=int,
                    help="Number of parallel runs to use in SMAC optimization.")
parser.add_argument("--random_seed", default=27, type=int,
                    help="Random seed used in optimization.")
parser.add_argument("--n_evaluations", default=30, type=int, 
                    help="Number of evaluations used in SMAC optimization.")
parser.add_argument("--cutoff_time", default=1000, type=int, 
                    help="Configuration will be terminated if it takes > cutoff_time to run.")

config = parser.parse_args()

##################################################################################################
# Helper functions                                                                               #
##################################################################################################

def get_files_as_ls(path_to_dir, extension='csv'):
    return glob.glob('{}/*.{}'.format(path_to_dir, extension))

def get_basename_from_ls(filepath_ls):
    return [os.path.basename(path) for path in filepath_ls]

def read_json_file(filename):
   with open(filename) as f_in:
       return(json.load(f_in))

##################################################################################################
# Main function                                                                                  #
##################################################################################################

def main():
    # Create output directory
    output_dir = LogUtils.create_new_directory(prefix='metalearning')    

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
    
    ##################################################################################################
    # Preprocessing                                                                                  #
    ##################################################################################################
    
    # get names of all raw datasets
    raw_data_filepath_ls = get_files_as_ls(config.raw_data_path, 'csv')
    raw_data_filename_ls = get_basename_from_ls(raw_data_filepath_ls)
    _logger.info("Managed to find {} raw datasets: {}".format(len(raw_data_filepath_ls), raw_data_filename_ls))
    
    # get names of all processed datasets
    processed_data_filepath_ls = get_files_as_ls(config.processed_data_path, 'csv')
    processed_data_filename_set = set(get_basename_from_ls(processed_data_filepath_ls))
    _logger.info("Managed to find {} processed datasets: {}".format(len(processed_data_filepath_ls), 
                                                                    processed_data_filename_set))
    
    # get all json files (required for preprocessing step)
    json_filepath_ls = get_files_as_ls(config.processed_data_path, 'json')
    json_filename_set = set(get_basename_from_ls(json_filepath_ls))
    
    # for each raw dataset, check if processed version exist, if no, preprocess it and save it
    for data_filename in raw_data_filename_ls:
        if data_filename not in processed_data_filename_set:
            # get file name without extension
            data_filename_no_ext, _ = os.path.splitext(data_filename)
            
            # get corresponding json filename, which tells us how to preprocess it
            json_filename = '{}.json'.format(data_filename_no_ext)
            
            # if we don't have a json file, we don't know how to preprocess this dataset
            if json_filename not in json_filename_set:
                _logger.info("Failed to find {}, so {} cannot be preprocessed.".format(json_filename,
                                                                                       data_filename))
                # ignore this dataset
                continue
            
            # read the json file as dictionary
            preprocess_config_dict = read_json_file('{}/{}'.format(config.processed_data_path, json_filename))
            _logger.info(preprocess_config_dict)
            
            # preprocess and then save it
            dataset_obj = PreprocessedDataset(**preprocess_config_dict)
            dataset_obj.save(config.processed_data_path, data_filename_no_ext)
            _logger.info('Saving proprocessed files.')
            
            # just for keeping track
            processed_data_filename_set.add(data_filename)
    
    # read names of processed datasets again
    processed_data_filepath_ls = get_files_as_ls(config.processed_data_path, 'csv')
    _logger.info("Preprocessing complete, there are now {} raw csv, {} processed csv".format(len(raw_data_filepath_ls), 
                                                                                             len(processed_data_filepath_ls)))
    
    ##################################################################################################
    # Meta Learning                                                                                  #
    ##################################################################################################
    
    # main loop of meta learning
    for i, dataset_path in enumerate(processed_data_filepath_ls, 1):
        # logging
        _logger.info("ITERATION {} of {}".format(i, len(processed_data_filepath_ls)))
        _logger.info("Optimizing hyperparameters on the dataset at: {}".format(dataset_path))
        
        # read processed dataset as dataframe
        dataset = pd.read_csv(dataset_path, header='infer', sep=',')
        dataset_np = dataset.to_numpy()
        dataset_basename = get_basename_from_ls([dataset_path])[0]
        
        # this dictionary will keep track of everything we need log
        records = {}
        records["dataset"] = dataset_basename
        
        # get raw dataset
        raw_dataset = pd.read_csv("{}/{}".format(config.raw_data_path, dataset_basename), 
                                  header='infer', sep=',')
        raw_dataset_np = raw_dataset.to_numpy()
        
        # calculate metafeatures
        records["numberOfInstances"] = Metafeatures.numberOfInstances(raw_dataset_np)
        records["numberOfFeatures"] = Metafeatures.numberOfFeatures(raw_dataset_np)
        
        # run autocluster
        autocluster = AutoCluster(logger=_logger)
        fit_config = {
            "X": dataset_np, 
            "cluster_alg_ls": [
                'KMeans', 'GaussianMixture', 'Birch', 
                'MiniBatchKMeans', 'AgglomerativeClustering', 'OPTICS', 
                'SpectralClustering', 'DBSCAN', 'AffinityPropagation', 'MeanShift'
            ], 
            "dim_reduction_alg_ls": [
                'TSNE', 'PCA', 'IncrementalPCA', 
                'KernelPCA', 'FastICA', 'TruncatedSVD'
            ],
            "n_evaluations": config.n_evaluations,
            "seed": config.random_seed, 
            "run_obj": 'quality', 
            "cutoff_time": config.cutoff_time, 
            "shared_model": False,
            "n_parallel_runs": config.n_parallel_runs,
            "evaluator": lambda X, y_pred: 
                            float('inf') if len(set(y_pred)) == 1 \
                            else -1 * silhouette_score(X, y_pred)  
        }
        smac_obj, opt_result = autocluster.fit(**fit_config)
        
        # save result
        records["trajectory"] = autocluster.get_trajectory()
        
        # log results
        _logger.info("Done optimizing on {}.".format(dataset_path))
        _logger.info("Record on ITERATION {}: \n{}".format(i, records))
        _logger.info("Done with ITERATION {}.".format(i))

if __name__ == '__main__':
    main()

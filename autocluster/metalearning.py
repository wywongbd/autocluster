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
from utils.metafeatures import MetafeatureMapper, calculate_metafeatures

from sklearn import datasets
from datetime import datetime
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

##################################################################################################
# Define parameters for script                                                                   #
##################################################################################################
parser = argparse.ArgumentParser()

# directories
parser.add_argument("--raw_data_path", type=str, default="../data/raw_data/",
                    help="Directory of raw datasets. Will be ignored if raw_data_path_ls is used.")
parser.add_argument("--raw_data_path_ls", default=[], nargs='+', type=str,
                    help="List of names of raw datasets to be processed. raw_data_path will be used if this is not provided.")
parser.add_argument("--processed_data_path", type=str, default='../data/processed_data/', 
                    help="Directory of processed datasets.")

# for logging
parser.add_argument("--log_dir_prefix", type=str, default='meta_learning', 
                    help='Prefix of directory')

# optimization
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
    if len(config.raw_data_path_ls) == 0:
        raw_data_filepath_ls = get_files_as_ls(config.raw_data_path, 'csv')
    else: 
        raw_data_filepath_ls = config.raw_data_path_ls
        
    raw_data_filename_ls = get_basename_from_ls(raw_data_filepath_ls)
    _logger.info("Managed to find {} raw datasets: {}".format(len(raw_data_filepath_ls), raw_data_filename_ls))
    
    # get all json files (required for preprocessing step)
    json_filepath_ls = get_files_as_ls(config.processed_data_path, 'json')
    json_filename_set = set(get_basename_from_ls(json_filepath_ls))
    
    # this list saves the paths of datasets which are ready to be processed
    ready_datasets_ls = []
    
    # for each raw dataset, check if processed version exist, if no, preprocess it and save it
    for raw_data_path, data_filename in zip(raw_data_filepath_ls, raw_data_filename_ls):
        # get file name without extension
        data_filename_no_ext, _ = os.path.splitext(data_filename)
        
        # get corresponding json filename, which tells us how to preprocess it
        # the json file also contains metadata, telling us how to calculate metafeatures
        json_filename = '{}.json'.format(data_filename_no_ext)
        
        # if we don't have a json file, we can't process it
        if json_filename not in json_filename_set:
            _logger.info("Failed to find {}, so {} cannot be used.".format(json_filename, data_filename))
            continue
        
        # save into ready_datasets_ls for later use
        json_file_path = '{}/{}.json'.format(config.processed_data_path, data_filename_no_ext)
        ready_datasets_ls.append((raw_data_path, json_file_path))
    
    # logging
    _logger.info("Going to perform metalearning on the following {} datasets: {}".format(len(ready_datasets_ls), 
                                                                                         ready_datasets_ls))
    
    ##################################################################################################
    # Meta Learning                                                                                  #
    ##################################################################################################
    
    # main loop of meta learning
    for i, pair in enumerate(ready_datasets_ls, 1):
        # logging
        _logger.info("ITERATION {} of {}".format(i, len(ready_datasets_ls)))
        
        # unpack paths 
        raw_dataset_path, json_file_path = pair
        _logger.info("Optimizing hyperparameters using these files: {}".format(pair))
        
        # read dataset as dataframe
        dataset = pd.read_csv(raw_dataset_path, header='infer', sep=',')
        dataset_basename = get_basename_from_ls([raw_dataset_path])[0]
        dataset_basename_no_ext, _ = os.path.splitext(raw_dataset_path)
        
        # this dictionary will keep track of everything we need log
        records = {}
        records["dataset"] = dataset_basename
        
        # get corresponding json filename, which tells us which columns are categorical and numerical
        json_filename = '{}.json'.format(dataset_basename_no_ext)
        json_file_dict = read_json_file(json_file_path)
        
        # logging
        _logger.info("general metafeatures: {}".format(MetafeatureMapper.getGeneralMetafeatures()))
        _logger.info("numeric metafeatures: {}".format(MetafeatureMapper.getNumericMetafeatures()))
        _logger.info("categorical metafeatures: {}".format(MetafeatureMapper.getCategoricalMetafeatures()))
        
        # calculate metafeatures
        metafeatures = calculate_metafeatures(dataset, json_file_dict, MetafeatureMapper.getAllMetafeatures())
        metafeatures_ls = list(metafeatures[0])
        records.update(dict(zip(MetafeatureMapper.getAllMetafeatures(), metafeatures_ls))) 
        _logger.info("metafeature values: {}".format(metafeatures_ls))
        
        # run autocluster
        autocluster = AutoCluster(logger=_logger)
        fit_params = {
            "df": dataset, 
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
            "run_obj": 'quality',
            "seed": config.random_seed,
            "cutoff_time": config.cutoff_time,
            "preprocess_dict": json_file_dict,
            "warmstart": False,
            "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
            "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
            "categorical_metafeatures": MetafeatureMapper.getCategoricalMetafeatures(),
        }
        smac_obj, opt_result = autocluster.fit(**fit_params)
        
        # save result
        records["trajectory"] = autocluster.get_trajectory()
        
        # log results
        _logger.info("Done optimizing on {}.".format(raw_dataset_path))
        _logger.info("Record on ITERATION {}: \n{}".format(i, records))
        _logger.info("Done with ITERATION {}.".format(i))

if __name__ == '__main__':
    main()

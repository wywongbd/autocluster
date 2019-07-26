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

from evaluators import get_evaluator
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
parser.add_argument("--raw_data_path_ls", default=[], nargs='+', type=str,
                    help="List of paths of raw datasets to be processed.")
parser.add_argument("--json_data_path_ls", default=[], nargs='+', type=str, 
                    help="List of json files required for preprocessing the raw datasets.")

# for logging
parser.add_argument("--log_dir_prefix", type=str, default='metalearning', 
                    help='Prefix of directory')

# optimization
parser.add_argument("--n_folds", default=3, type=int,
                    help="Number of folds used in k-fold cross validation during evaluation step of SMAC optimization.")
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

def get_basename_with_no_ext(path):
    basename = os.path.basename(path)
    basename_no_ext, _ = os.path.splitext(basename)
    return basename_no_ext

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
    
    ##################################################################################################
    # Matching raw data with corresponding JSON file                                                 #
    ##################################################################################################
    basename_to_raw_dict = {
        get_basename_with_no_ext(path): path for path in config.raw_data_path_ls
    }
    basename_to_json_dict = {
        get_basename_with_no_ext(path): path for path in config.json_data_path_ls
    }
    
    # take the intersection of two dictionaries
    common_basenames = set(basename_to_raw_dict.keys()) & set(basename_to_json_dict.keys())
    
    # filter out non-common basenames
    basename_to_raw_dict = {k: v for k, v in basename_to_raw_dict.items() if k in common_basenames}
    basename_to_json_dict = {k: v for k, v in basename_to_json_dict.items() if k in common_basenames}
    
    # combine the two dictionaries
    merged_dict = {k: (v, basename_to_json_dict[k]) for k, v in basename_to_raw_dict.items()}
    
    # logging
    _logger.info("Going to perform metalearning on the following {} datasets: {}".format(len(merged_dict), 
                                                                                         list(merged_dict.values())))
    ##################################################################################################
    # Meta Learning                                                                                  #
    ##################################################################################################
    # main loop of meta learning
    for i, pair in enumerate(merged_dict.values(), 1):
        # logging
        _logger.info("ITERATION {} of {}".format(i, len(merged_dict)))
    
        # unpack paths 
        raw_dataset_path, json_file_path = pair
        _logger.info("Optimizing hyperparameters using these files: {}".format(pair))
        
        try:
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
            json_file_dict = {k: v for k, v in json_file_dict.items() 
                              if k in ["numeric_cols", "categorical_cols", "ordinal_cols", "y_col", "ignore_cols"]}
            
            # for safety reasons
            json_file_dict["numeric_cols"] = json_file_dict.get("numeric_cols", [])
            json_file_dict["categorical_cols"] = json_file_dict.get("categorical_cols", [])
            json_file_dict["ordinal_cols"] = json_file_dict.get("ordinal_cols", {})
            json_file_dict["y_col"] = json_file_dict.get("y_col", None)
            json_file_dict["ignore_cols"] = json_file_dict.get("ignore_cols", [])

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
                    'KernelPCA', 'FastICA', 'TruncatedSVD',
                    'NullModel'
                ],
                "n_evaluations": config.n_evaluations,
                "run_obj": 'quality',
                "seed": config.random_seed,
                "cutoff_time": config.cutoff_time,
                "preprocess_dict": json_file_dict,
                "evaluator": get_evaluator(evaluator_ls = ['silhouetteScore'], 
                                           weights = [], clustering_num = None, 
                                           min_proportion = .01),
                "n_folds": config.n_folds,
                "warmstart": False,
                "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
                "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
                "categorical_metafeatures": MetafeatureMapper.getCategoricalMetafeatures(),
            }
            result_dict = autocluster.fit(**fit_params)

            # save result
            records["trajectory"] = autocluster.get_trajectory()
            metafeatures_ls = list(result_dict["metafeatures"][0])
            records.update(dict(zip(result_dict["metafeatures_used"], metafeatures_ls))) 

            # log results
            _logger.info("Done optimizing on {}.".format(raw_dataset_path))
            _logger.info("Record on ITERATION {}: \n{}".format(i, records))
            _logger.info("Done with ITERATION {}.".format(i))
        
        except Exception as e:
            # logging
            _logger.info("Error message: {}".format(str(e)))
            _logger.info("Traceback: {}".format(traceback.format_exc()))
            _logger.info("ITERATION {} of {} FAILED! ".format(i, len(merged_dict)))

if __name__ == '__main__':
    main()

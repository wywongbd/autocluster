import glob
import os
import sys
import argparse
import pathlib
import logging

from sklearn import datasets
from datetime import datetime
from log_helper.log_helper import LogHelper
from utils.logutils import LogUtils
from autocluster import AutoCluster
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

##################################################################################################
# Define parameters for script                                                                   #
##################################################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--raw_data_path", type=str, default="../data/raw_data/",
                    help="Directory of raw datasets.")
parser.add_argument("--processed_data_path", type=str, default='../data/processed_data/', 
                    help="Directory of processed datasets.")
parser.add_argument("--log_dir_prefix", type=str, default='meta_learning', help='Prefix of directory')

parser.add_argument("--n_parallel_runs", default=3, type=int,
                    help="Number of parallel runs to use in SMAC optimization.")

config = parser.parse_args()

##################################################################################################
# Main function                                                                                  #
##################################################################################################

def main():
    ##################################################################################################
    # Setup logger and output dir                                                                    #
    ##################################################################################################
    
    # Create output directory
    output_dir = LogUtils.create_new_directory(prefix='metalearning')    

    # Setup logger
    LogHelper.setup(log_path='{}/meta.log'.format(output_dir), log_level=logging.INFO)
    _logger = logging.getLogger(__name_)





    
    
    #get the filenames from the ../data directory
    file_directory = config.raw_data_path
    file_list = [join(file_directory,f) for f in listdir(file_directory) if isfile(join(file_directory, f))]
    print (file_list)

    # Log all parameters
    _logger_path = logging.getLoggerClass().root.handlers[0].baseFilename
    _logger.info("Meta-learning parameters: {}".format(vars(config)))
    _logger.info("Log file at {}" .format(_logger_path))

    
    X = datasets.load_iris().data
    autocluster = AutoCluster(logger=_logger)
    smac_obj, opt_result = autocluster.fit(X, cluster_alg_ls=['KMeans'], 
                                           dim_reduction_alg_ls=[],
                                           n_evaluations=40, seed=27, run_obj='quality', cutoff_time=10, 
                                           shared_model=True, n_parallel_runs = 3,
                                           evaluator=lambda X, y_pred: float('inf') if len(set(y_pred)) == 1 \
                                                    else -1 * silhouette_score(X, y_pred)  
                                          ) 

if __name__ == '__main__':
    main()

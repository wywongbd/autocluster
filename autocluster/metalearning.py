import glob
import os
import sys
import argparse
import pathlib
import logging

from datetime import datetime
from log_helper.log_helper import LogHelper
from utils.logutils import LogUtils

from autocluster import AutoCluster
from algorithms import algorithms
from build_config_space import build_config_space
from utils.clusterutils import ClusterUtils
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn import datasets
import csv
from os import listdir
from os.path import isfile, join

import numpy as np

##################################################################################################
# Define parameters for script                                                                   #
##################################################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--raw_data_path", type=str, default="../data/raw_data/",
                    help="Directory of raw datasets.")
parser.add_argument("--processed_data_path", type=str, default='../data/processed_data_path/', 
                    help="Directory of processed datasets.")

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
    output_dir = LogUtils.create_new_directory(prefix='meta_learner')    

    # Setup logger
    LogHelper.setup(log_path='{}/backtesting.log'.format(output_dir), log_level=logging.INFO)
    _logger = logging.getLogger(__name__)

    # Log all paremeters
    _logger.info("Meta learning parameters: {}".format(vars(config)))
    
    
    
    #get the filenames from the ../data directory
    file_directory = config.raw_data_path
    #file_list = [join(file_directory,f) for f in listdir(file_directory) if isfile(join(file_directory, f))]
    #print (file_list)
    
    #TODO preprocess the data into something suitable for clustering

    for file in ['../data/credit_card_dataset.csv']:
        #load the data from csv
        with open(file, 'r') as f:
            reader = csv.reader(f)
            X = list(reader)

        del X[0]

        X = [[float(numStr) if numStr else 0 for numStr in sublist[1:]] for sublist in X]
        X = np.array(X)

        #preform preprocessing

        #preform metafeature identification

        ClusterUtils.visualize_sample_data(X)
        autocluster = AutoCluster()
        smac_obj, opt_result = autocluster.fit(X, cluster_alg_ls=['GaussianMixture'], 
                                               dim_reduction_alg_ls=['TSNE'],
                                               n_evaluations=50, seed=27, run_obj='quality', cutoff_time=10, 
                                               shared_model=True, n_parallel_runs = 3,
                                               evaluator=lambda X, y_pred: float('inf') if len(set(y_pred)) == 1 \
                                                        else -1 * silhouette_score(X, y_pred)  
        #                                                    else davies_bouldin_score(X, y_pred)
                                              )

        predictions = autocluster.predict(X)
        np.unique(predictions)
        
    _logger.info("{}".format(opt_result))
        

if __name__ == '__main__':
    main()
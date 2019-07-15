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
parser.add_argument("--processed_data_path", type=str, default='../data/processed_data/', 
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
    
    

if __name__ == '__main__':
    main()
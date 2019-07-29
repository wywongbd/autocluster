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

from autocluster import AutoCluster
from log_helper.log_helper import LogHelper
from utils.logutils import LogUtils
from utils.metafeatures import MetafeatureMapper, calculate_metafeatures
from evaluators import get_evaluator
from sklearn.metrics.cluster import v_measure_score

from sklearn import datasets
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("--random_seed", default=27, type=int,
                    help="Random seed used in optimization.")
parser.add_argument("--n_evaluations", default=100, type=int, 
                    help="Number of evaluations used in SMAC optimization.")

config = parser.parse_args()

def main():
    # Create output directory
    output_dir = LogUtils.create_new_directory(prefix='test_evaluator_combinations')    

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
    
    # load data
    df = pd.DataFrame(datasets.load_digits()['data'])
    
    # variables for keeping track
    top10_v_scores = []
    top10_evaluator_weights = []
    prediction_num = 0
    
    # main loop
    ITERATION = 1
    for silhouette in range(0, 11):
        for davies in range(0, 11 - silhouette):
            _logger.info("Running ITERATION {} of 66.".format(ITERATION))
            try:
                calinski = 10 - silhouette - davies
                cluster = AutoCluster()
                fit_params = {
                    "df": df, 
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
                    "cutoff_time": 20,
                    "preprocess_dict": {
                        "numeric_cols": list(range(64)),
                        "categorical_cols": [],
                        "ordinal_cols": [],
                        "y_col": []
                    },
                    "evaluator": get_evaluator(evaluator_ls = ['silhouetteScore', 'daviesBouldinScore', 'calinskiHarabaszScore'], 
                                               weights = [silhouette, davies, calinski], clustering_num = None, 
                                               min_proportion = .01),
                    "n_folds": 3,
                    "warmstart": True,
                    "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
                    "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
                    "categorical_metafeatures": [],
                }
                result_dict = cluster.fit(**fit_params)
                predictions = cluster.predict(df, plot=False)
                v_score = v_measure_score(predictions, datasets.load_digits()['target'])

                if prediction_num < 10:
                    for i in range(prediction_num):
                        if v_score > top10_v_scores[i]:
                            top10_v_scores.insert(i, v_score)
                            top10_evaluator_weights.insert(i, [silhouette, davies, calinski])
                            break
                    if len(top10_v_scores) == prediction_num:
                        top10_v_scores.append(v_score)
                        top10_evaluator_weights.append([silhouette, davies, calinski])
                    prediction_num += 1
                    print("\nModel chosen with weights = ")
                    print([silhouette, davies, calinski])
                    print("v_score = ")
                    print(v_score)
                    print("\n")
                else:
                    for i in range(10):
                        if v_score > top10_v_scores[i]:
                            top10_v_scores.insert(i, v_score)
                            top10_evaluator_weights.insert(i, [silhouette, davies, calinski])
                            del(top10_v_scores[10])
                            del(top10_evaluator_weights[10])
                            print("\nModel chosen with weights = ")
                            print([silhouette, davies, calinski])
                            print("v_score = ")
                            print(v_score)
                            print("\n")
                            break
            except Exception as e:
                # logging
                _logger.info("Error message: {}".format(str(e)))
                _logger.info("Traceback: {}".format(traceback.format_exc()))
            
            _logger.info("Done with ITERATION {} of 66.".format(ITERATION))
            ITERATION += 1
    
    results = {
        "top10_v_scores": top10_v_scores,
        "top10_evaluator_weights": top10_evaluator_weights
    }
    _logger.info("All iterations completed, results: {}".format(results))

if __name__ == '__main__':
    main()
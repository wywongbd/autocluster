import json
import glob
import os
import sys
import argparse
import pathlib
import logging
import pandas as pd

# change directory
sys.path.append("../")

import autocluster
from autocluster import Decoder, LogHelper, LogUtils

##################################################################################################
# Define parameters for script                                                                   #
##################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--log_filepath_ls", default=[], nargs='+', type=str,
                    help="List of log files to process.")
parser.add_argument("--metafeatures_filename", type=str, default='metafeatures_table', 
                    help="Name of metafeatures table (to be generated).")
parser.add_argument("--datasets_dir", default='datasets', type=str,
                    help="Which folder does this metaknowledge belongs to?")

config = parser.parse_args()

##################################################################################################
# Helper functions                                                                               #
##################################################################################################

def collect_and_save_metaknowledge(log_filepath_ls, 
                                   metafeatures_filename='metafeatures_table',
                                   datasets_dir='datasets',
                                   logger=None
                                  ):
    # logging function
    def log(string):
        if logger is None:
            print(string)
        else:
            logger.info(string)
    
    # decode metaknowledge from log files
    metadata = {}
    for path in log_filepath_ls:
        metadata.update(Decoder.decode_log_file(path=path))
        
    log("Metaknowledge extracted from following datasets: {}".format(list(metadata.keys())))
    
    # magic stuff
    is_primitive_or_none = lambda value: type(value) in (int, str, bool, float) or value is None
    
    # this thing will be turned into a dataframe later
    metadata_ls = [
        {k: v for k, v in metadata[d].items() if is_primitive_or_none(v)} for d in metadata
    ]
    
    # get the set of all keys
    allkeys = set().union(*metadata_ls)
    
    log("The metafeatures table will contain the following columns: {}".format(allkeys))
    
    # fill None for missing keys
    for d in metadata_ls:
        missingkeys = allkeys.difference(set(d.keys()))
        for k in missingkeys:
            d[k] = None
            
    # save the metafeatures table
    metafeatures_table = pd.DataFrame.from_dict(metadata_ls)
    metafeatures_table.to_csv('{}/{}.csv'.format('metaknowledge', metafeatures_filename), 
                              encoding='utf-8', 
                              index=False)
    
    log("Saved metafeatures table as csv file.")
    
    # create directory if doesn't exist
    if not os.path.exists('metaknowledge/{}'.format(datasets_dir)):
        pathlib.Path('metaknowledge/{}'.format(datasets_dir)).mkdir(parents=True, exist_ok=True)
    
    # save the metaknowledge of each dataset as csv for retrieval of runhistory
    for d in metadata:
        string = json.dumps(metadata[d])
        d_no_ext, _ = os.path.splitext(d)
        print(string,  
              file=open('{}/{}/{}.json'.format('metaknowledge', datasets_dir, d_no_ext), 'w'))
        
    log("Saved metaknowledge of each dataset.")
    
##################################################################################################
# Main function                                                                                  #
##################################################################################################
    
def main():
    # Create output directory
    output_dir = LogUtils.create_new_directory(prefix='metaknowledge')    

    # Setup logger
    LogHelper.setup(log_path='{}/meta.log'.format(output_dir), log_level=logging.INFO)
    _logger = logging.getLogger(__name__)
    _logger_path = logging.getLoggerClass().root.handlers[0].baseFilename
    _logger.info("Log file location: {}".format(_logger_path))
    
    # log all arguments passed into this script
    _logger.info("Script arguments: {}".format(vars(config)))
    
    # collect and save metaknowledge
    collect_and_save_metaknowledge(log_filepath_ls=config.log_filepath_ls, 
                                   metafeatures_filename=config.metafeatures_filename,
                                   datasets_dir=config.datasets_dir
                                  )
    
if __name__ == '__main__':
    main()
        
            
    
        
    
        
        
        
    
    
    
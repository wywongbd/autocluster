import json
import os
import pathlib
import pickle
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

class KDTreeWarmstarter(object):
    @staticmethod
    def load_from_dir(name):
        """
        Returns a KDTreeWarmstarter object
        """
        path = 'metaknowledge/models/{}'.format(name)
        
        # load table
        table = pd.read_csv('{}/{}.csv'.format(path, 'table'), 
                            sep=',', header='infer')
        
        # load model
        with open('{}/{}.pickle'.format(path, 'model'), 'rb') as f:
            tree = pickle.load(f)
            
        # load scaler
        with open('{}/{}.pickle'.format(path, 'scaler'), 'rb') as f:
            scaler = pickle.load(f)      
        
        # construct object
        obj = KDTreeWarmstarter()
        obj.metafeatures = list(table.columns)
        obj.table = table
        obj.model = tree
        obj.scaler = scaler
        
        return obj
    
    def __init__(self, metafeatures=["numberOfInstances", "logNumberOfInstances"]):
        self.metafeatures = metafeatures
        self.model = None
        self.table = None
        self.scaler = StandardScaler()
    
    def fit(self, metafeatures_table_path='metaknowledge/metafeatures_table.csv'):
        """
        Fit a KDTree model
        """
        # read metafeatures table as dataframe
        self.table = pd.read_csv(metafeatures_table_path, sep=',', header='infer')
        
        # only consider columns that we're interested
        self.table = self.table[self.metafeatures + ['dataset']]
        
        # remove rows with NaN 
        self.table = self.table.dropna()
        
        # this is the dataframe we want to fit our model on
        table_without_dataset = self.table.drop(columns=['dataset'])
        table_without_dataset_np = table_without_dataset.to_numpy()
        
        # train the scaler
        self.scaler.fit(table_without_dataset_np)
        
        # fit KDTree model
        self.model = KDTree(self.scaler.transform(table_without_dataset_np), 
                            leaf_size=5)
    
    def query(self, features, neighbor_k=3, top_k=10, datasets_dir='silhouette'):
        """
        Returns a list of configurations (in dictionary form)
        """
        # scale the features
        features = self.scaler.transform(features)
        
        # query for indices of neighbor_k most similar dataset
        _, idx_ls = self.model.query(features, k=neighbor_k) 
        
        # get the names of datasets 
        datasets = list(self.table['dataset'][idx_ls[0]])
        
        # helper function
        def read_json_file(filename):
            with open(filename) as f_in:
                return(json.load(f_in))
            
        # this list will be returned
        config_ls = []
        
        # add configurations to list
        for name in datasets:
            name_no_ext, _ = os.path.splitext(name)
            dic = read_json_file('{}/{}/{}.json'.format('metaknowledge', datasets_dir, name_no_ext))
            config_ls.extend(dic['runhistory'][0 : top_k])
            
        return config_ls
        
    def save(self, name):
        output_dir = 'metaknowledge/models/{}'.format(name)
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        # save table
        self.table.to_csv('{}/{}.csv'.format(output_dir, 'table'), 
                          encoding='utf-8', index=False)

        # save model
        with open('{}/{}.pickle'.format(output_dir, 'model'), 'wb') as f:
            pickle.dump(self.model, f) 
            
        # save scaler
        with open('{}/{}.pickle'.format(output_dir, 'scaler'), 'wb') as f:
            pickle.dump(self.scaler, f) 
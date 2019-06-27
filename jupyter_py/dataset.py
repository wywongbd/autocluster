import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 

class Dataset(object):
    def __init__(self, path, y_col, train_size=0.5, val_size=0.25, test_size=0.25, 
                 numeric_cols=[], categorical_cols=[], ignore_cols=[], classification=True):
        """
        --------------------------------------------------------------
        Arguments:
        --------------------------------------------------------------
        path - string type
        y_col - name of target variable column
        numerical_cols - list of column names of numerical features
        categorical_cols - list of column names of categorical features
        train_size + val_size + test_size must be equal to 1
        classification - set to False if you are doing regression
        """
        # read csv file as dataframe
        self.data_pd = pd.read_csv(path, sep= ',', header='infer')
        
        # ignore columns that are not relevant
        self.data_pd = self.data_pd.drop(columns=ignore_cols)
        
        # split dataset into features and output
        self.X_pd, self.y_pd = Dataset.feature_target_split(self.data_pd, y_col)
        
        # cast feature columns to categorical, numerical
        self.categorical_mappings = {}

        if len(categorical_cols) > 0:
            self.X_pd[categorical_cols] = self.X_pd[categorical_cols].astype('category')
            
            for col in categorical_cols:
                # save the mapping
                self.categorical_mappings[col] = dict(enumerate(self.X_pd[col].cat.categories))
                # map categories to integers
                self.X_pd[col] = self.X_pd[col].cat.codes
            
        if len(numeric_cols) > 0:
            self.X_pd[numeric_cols] = self.X_pd[numeric_cols].astype(float)
            
        if classification:
            self.y_pd[y_col] = self.y_pd[y_col].astype('category')
            self.categorical_mappings[y_col] = dict(enumerate(self.y_pd[y_col].cat.categories))
            self.y_pd = pd.DataFrame(self.y_pd[y_col].cat.codes)
        else:
            self.y_pd[y_col] = self.y_pd[y_col].astype(float)
        
        # convert dataframe to numpy matrix
        self.X, self.y = self.X_pd.to_numpy(), self.y_pd.to_numpy()
        
        # save train, validation and test proportions
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        
        # split into train, validation and test data
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
        Dataset.split_data(self.X, self.y, train_size, val_size, test_size)
        
    def print_shapes(self):
        d = {
            'X_train': self.X_train.shape,
            'y_train': self.y_train.shape,
            'X_val': self.X_val.shape,
            'y_val': self.y_val.shape,
            'X_test': self.X_test.shape,
            'y_test': self.y_test.shape,
        }
        print("{}".format(d))
        
    @staticmethod
    def feature_target_split(data, y_col):
        return data.drop(columns=y_col), pd.DataFrame(data[y_col])
        
    @staticmethod
    def split_data(X, y, train_size, val_size, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = val_size + test_size, 
                                                            random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                          test_size = val_size, 
                                                          random_state=1)  
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    
    
    
    

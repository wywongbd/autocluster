import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class Dataset(object):
    def __init__(self, path, y_col, header='infer', train_size=0.6, test_size=0.4, 
                 numeric_cols=[], 
                 categorical_cols=[],
                 ordinal_cols={},
                 ignore_cols=[], 
                 classification=True):
        """
        --------------------------------------------------------------
        Arguments:
        --------------------------------------------------------------
        path - string type
        y_col - name of target variable column
        ignore_cols - list of columns to be ignored
        train_size + test_size must be equal to 1
        classification - set to False if you are doing regression
        
        numeric_cols - list of column names of numerical features
        categorical_cols - list of column names of categorical features
        ordinal_cols - a dictionary where each key is a column name, each value is a list of ordinal_values (ordered)
        """
        # read csv file as dataframe
        self.data_df_raw = pd.read_csv(path, sep= ',', header=header)
        
        # ignore columns that are not relevant
        self.data_df = self.data_df_raw.drop(columns=ignore_cols)
        
        # split dataset into features and output
        self.X_df, self.y_df = Dataset.feature_target_split(self.data_df, y_col)
        
        # save the transformation of categorical and ordinal cols
        self.encoded_cols = {}
        self.encodings = {}
        
        # one hot encode categorical columns
        for col in categorical_cols:
            self.X_df[col] = self.X_df[col].astype('category')
            self.encodings[col] = dict(enumerate(self.X_df[col].cat.categories))
            integer_encoded_col = self.X_df[col].cat.codes
            one_hot_encoded_col = pd.get_dummies(integer_encoded_col, prefix=col)
            self.encoded_cols[col] = one_hot_encoded_col
            
        # encode ordinal columns
        for col in ordinal_cols:
            self.encodings[col] = dict(zip(ordinal_cols[col], range(len(ordinal_cols[col]))))
            self.encoded_cols[col] = pd.DataFrame(self.X_df[col].replace(to_replace=self.encodings[col]), columns=[col])
        
        # merge the encoded columns
        self.X_df_encoded, self.y_df_encoded = pd.DataFrame(self.X_df[numeric_cols]), pd.DataFrame(self.y_df)
        
        for col in self.encoded_cols:
            self.X_df_encoded[self.encoded_cols[col].columns] = self.encoded_cols[col]
            
        # encode the output column, if necessary
        if classification:
            self.y_df_encoded[y_col] = self.y_df[y_col].astype('category')
            self.encodings[y_col] = dict(enumerate(self.y_df_encoded[y_col].cat.categories))
            self.y_df_encoded[y_col] = self.y_df_encoded[y_col].cat.codes
        else:
            self.y_df_encoded[y_col] = self.y_df[y_col].astype('float')
        
        # save train and test proportions
        self.train_size, self.test_size = train_size, test_size
        
        # df to numpy
        self.X, self.y = self.X_df_encoded.to_numpy(), self.y_df_encoded.to_numpy()
        
        # split into train, validation and test data
        self.X_train, self.y_train, self.X_test, self.y_test = Dataset.split_data(self.X, self.y, train_size, test_size)
        
    def print_shapes(self):
        d = {
            'X_train': self.X_train.shape,
            'y_train': self.y_train.shape,
            'X_test': self.X_test.shape,
            'y_test': self.y_test.shape,
        }
        print("{}".format(d))
    
    @staticmethod
    def feature_target_split(data, y_col):
        return data.drop(columns=y_col), pd.DataFrame(data[y_col])
        
    @staticmethod
    def split_data(X, y, train_size, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = test_size, 
                                                            random_state=1)
        return X_train, y_train, X_test, y_test
    
    
    
    

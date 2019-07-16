rom collections import defaultdict, OrderedDict, deque
import copy
import sys

import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
import sklearn
# TODO use balanced accuracy!
import sklearn.metrics
import sklearn.model_selection
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Metafeatures(object):
    
    @staticmethod
    def calc_NumberOfInstances(self, X):
        return float(X.shape[0])
    
    @staticmethod
    def calc_NumberOfFeatures(self, X):
        return float(X.shape[1])
    
    @staticmethod
    def calc_NumberOfNumericFeatures(self, X):
        #checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is not str:
                count += 1
        return count
    
    @staticmethod
    def calc_NumberOfCategoricalFeatures(self, X):
        #checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is str:
                count += 1
        return count
    
    @staticmethod
    def calc_NumberOfInstances(self, X):
        return float(X.shape[0])
    
    
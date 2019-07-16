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

    def __init__(self):
        self._dataset = None
        self._numberOfInstance = None
        self._numberOfFeatures = None
        self._numberOfNumericFeatures = None
        self._numberOfCategoricalFeatures = None

    def calc_MetaFeatures(dataset):
        _dataset = dataset
        _numberOfInstance = calc_NumberOfInstances(_dataset)
        _numberOfFeatures = calc_NumberOfFeatures(_dataset)
        _numberOfNumericFeatures = calc_NumberOfNumericFeatures(_dataset)
        _numberOfCategoricalFeatures = calc_NumberOfCategoricalFeatures(_dataset)


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
        #checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is not str:
                count += 1
        return count

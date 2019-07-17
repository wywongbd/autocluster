import numpy as np
import scipy.stats

# all functions are based off of the implementation in autosklearn

class Metafeatures(object):
    @staticmethod
    def numberOfInstances(X):
        return len(X)
    
    @staticmethod
    def logNumberOfInstances(X):
        return np.log(numberOfInstances(X))

    @staticmethod
    def numberOfFeatures(X):
        return X.shape[1]
    
    @staticmethod
    def logNumberOfFeatures(X):
        return np.log(numberOfFeatures(X))

    @staticmethod
    def numberOfNumericFeatures(X):
        # checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is not str:
                count += 1
        return count

    @staticmethod
    def numberOfCategoricalFeatures(X):
        # checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is str:
                count += 1
        return count

    #returns True is any data is missing
    @staticmethod
    def missingValues(X):
        missing = ~np.isfinite(X)
        return missing

    @staticmethod
    def datasetRatio(X):
        return float(numberOfFeatures(X)) /\
            float(numberOfInstances(X))
    
    @staticmethod
    def kurtosisses(X):
        if(missingValues(X)):
            print("Error in Kurtossis calculation, dataset has missing values")
            return False
        
        kurts = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                kurts.append(scipy.stats.kurtosis(X[:, i]))
        return kurts
    
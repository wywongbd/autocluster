import numpy as np
import scipy.stats as scStat

# all functions are based off of the implementation in autosklearn

class Metafeatures(object):
    # type of X is ndarray
    # X has no headers
    
    @staticmethod
    def numberOfInstances(X):
        return len(X)
    
    # X shouldnt have labels column
    @staticmethod
    def logNumberOfInstances(X):
        return np.log(numberOfInstances(X))
    

    # X shouldnt have labels column
    @staticmethod
    def numberOfFeatures(X):
        return X.shape[1]
    
    @staticmethod
    def logNumberOfFeatures(X):
        return np.log(numberOfFeatures(X))
    
    
    # X should have only labels column
    @staticmethod
    def numberOfClasses(X):
        return len(np.unique(X))
    
    
    @staticmethod
    def numberOfMissingValues(X):
        count = 0
        for sublist in X:
            for value in sublist:
                if not value:
                    count++
        return count

    @staticmethod
    def missingValuesRatio(X):
        num = len(X) * len(X[0])
        return numberOfMissingValues(X) / num
    
    
    # X only have numerical columns
    @staticmethod
    def minSkewness(X):
        skewness = scStat.skew(X)
        return min(skewness)
    
    # X only have numerical columns
    @staticmethod
    def maxSkewness(X):
        skewness = scStat.skew(X)
        return max(skewness)
    
    # X only have numerical columns
    @staticmethod
    def medianSkewness(X):
        skewness = scStat.skew(X)
        return np.median(skewness)
    
    # X only have numerical columns
    @staticmethod
    def meanSkewness(X):
        skewness = scStat.skew(X)
        return np.mean(skewness)
    
    # X only have numerical columns
    @staticmethod
    def firstQuartileSkewness(X):
        skewness = scStat.skew(X)
        return np.percentile(skewness, 25)
    
    # X only have numerical columns
    @staticmethod
    def thirdQuartileSkewness(X):
        skewness = scStat.skew(X)
        return np.percentile(skewness, 75)
    
    
    # X only have numerical columns
    @staticmethod
    def minKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        return min(kurtosis)
    
    # X only have numerical columns
    @staticmethod
    def maxKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        return max(kurtosis)
    
    # X only have numerical columns
    @staticmethod
    def medianKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        return np.median(kurtosis)
    
    # X only have numerical columns
    @staticmethod
    def meanKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        return np.mean(kurtosis)
    
    # X only have numerical columns
    @staticmethod
    def firstQuartileKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        return np.percentile(kurtosis, 25)
    
    # X only have numerical columns
    @staticmethod
    def thirdQuartileKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        return np.percentile(kurtosis, 75)
    
    
    # X only have numerical columns
    @staticmethod
    def minCorrelation(X):
        corr = np.corrcoef(X.T)
        return np.min(corr)
    
    # X only have numerical columns
    @staticmethod
    def maxCorrelation(X):
        corr = np.corrcoef(X.T)
        for i in range(len(corr)):
            corr[i][i] = 0
        return np.man(corr)
    
    
#     @staticmethod
#     def numberOfNumericFeatures(X):
#         # checks if the item is a string
#         count = 0
#         for item in X[1]:
#             if type(item) is not str:
#                 count += 1
#         return count

#     @staticmethod
#     def numberOfCategoricalFeatures(X):
#         # checks if the item is a string
#         count = 0
#         for item in X[1]:
#             if type(item) is str:
#                 count += 1
#         return count

#     #returns True is any data is missing
#     @staticmethod
#     def missingValues(X):
#         missing = ~np.isfinite(X)
#         return missing

#     @staticmethod
#     def datasetRatio(X):
#         return float(numberOfFeatures(X)) /\
#             float(numberOfInstances(X))
    
#     @staticmethod
#     def kurtosisses(X):
#         if(missingValues(X)):
#             print("Error in Kurtossis calculation, dataset has missing values")
#             return False
        
#         kurts = []
#         for i in range(X.shape[1]):
#             if not categorical[i]:
#                 kurts.append(scipy.stats.kurtosis(X[:, i]))
#         return kurts
    
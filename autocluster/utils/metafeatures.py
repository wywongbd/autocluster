import numpy as np
import scipy.stats
import sklearn.decomposition

# this class contains the static function we use to calculate metafeatures
class Metafeatures(object):
    
    @staticmethod
    def isCategorical(item): # in the future this function may change slightly if the input changes
        if type(item) is str:
            return True
        else:
            return False

    @staticmethod
    def isOrdinal(item):
        return True
    
    @staticmethod
    def isNumerical(item):
        if type(item) is str:
            return False
        else:
            return True
    
    @staticmethod
    def numberOfInstances(X):
        return float(X.shape[0])
    
    @staticmethod
    def logNumberOfInstances(X):
        return np.log(numberOfInstances(X))
    
    @staticmethod
    def numberOfFeatures(X):
        return float(X.shape[1])
    
    @staticmethod
    def logNumberOfFeatures(X):
        return np.log(numberOfFeatures(X))
    
    #currently this is used to check for mistaken input 
    # in the future maybe we can generalize for dataset with missing data?
    @staticmethod
    def isMissingFeatures(X):
        missing = ~np.isfinite(X)
        return missing
    
    @staticmethod
    def numberOfNumericFeatures(X):
        #checks if the item is a string
        count = 0
        for item in X[1]:
            if isNumerical(item):
                count += 1
        return count
    
    @staticmethod
    def numberOfCategoricalFeatures(X):
        #checks if the item is a string
        count = 0
        for item in X[1]:
            if isCategorical(item):
                count += 1
        return count
    
    @staticmethod
    def ratioNumericalToNominal(X):
        num_categorical = float(numberOfCategoricalFeatures(X))
        num_numerical = float(numberOfNumericFeatures(X))
        if num_categorical == 0.0:
            return 0.
        else: 
            return num_numerical / num_categorical
    
    @staticmethod
    def ratioNominalToNumerical(X):
        num_categorical = float(numberOfCategoricalFeatures(X))
        num_numerical = float(numberOfNumericFeatures(X))
        if num_numerical == 0.0:
            return 0.
        else: 
            return num_categorical/num_numerical
    
    @staticmethod
    def datasetRatio(X):
        return float(numberOfFeatures(X)) /\
            float(numberOfInstances(X))
    
    @staticmethod
    def logDatasetRatio(X):
        return np.log(datasetRatio(X))
    
    @staticmethod
    def Kurtosisses(X):
        if isMissingFeatures(X):
            print('Error in calculating Kurtosis, data is missing values')
            return None
    
        kurts = []
        for i in range(X.shape[1]):
            if not isCategorical(i):
                kurts.append(scipy.stats.kurtosis(X[:, i]))
        return kurts
        
    @staticmethod
    def Skewnesses(X):
        if isMissingFeatures(X):
            print('Error in calculating Skewness, data is missing values')
            return None
        
        skews = []
        for i in range(X.shape[1]):
            if not isCategorical(i):
                skews.append(scipy.stats.skew(X[:, i]))
        return skews
    
    # todo we may want to calculate some more nuanced metrics for kurtosis and skewness in the future.

    # todo not sure if landmarking can be done without labels, to look into and maybe implement later
    
    @staticmethod
    def PCA(X):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X[indices])
                return pca
            except LinAlgError as e:
                pass
        print("Failed to compute PCA")
        #self.logger.warning("Failed to compute a Principle Component Analysis")
        return None
    
    @staticmethod
    def PCAFractionOfComponentsFor95PercentVariance(X):
        pca_ = PCA(X)
        if pca_ is None:
            return np.NaN
        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1
        return float(idx)/float(X.shape[1])
    
    @staticmethod
    def PCAKurtosisFirstPC(X):
        pca_ = PCA(X)
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0]
    
    @staticmethod
    def PCASkewnessFirstPC(X):
        pca_ = helper_functions.get_value("PCA")
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        skewness = scipy.stats.skew(transformed)
        return skewness[0]
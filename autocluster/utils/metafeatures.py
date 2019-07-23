import numpy as np
import scipy.stats as scStat
import sklearn.decomposition
from sklearn.impute import SimpleImputer

# all functions are based off of the implementation in autosklearn
# this class contains the static function we use to calculate metafeatures

class Metafeatures(object):
    ########################
    # type of X is ndarray #               
    # X has no headers     #
    ########################
    
#     @staticmethod
#     def isCategorical(item): # in the future this function may change slightly if the input changes
#         if type(item) is str:
#             return True
#         else:
#             return False

#     @staticmethod
#     def isOrdinal(item):
#         return True
    
#     @staticmethod
#     def isNumerical(item):
#         if type(item) is str:
#             return False
#         else:
#             return True
    
    # X shouldnt have labels column
    @staticmethod
    def numberOfInstances(X):
        return float(X.shape[0])
    
    # X shouldnt have labels column
    @staticmethod
    def logNumberOfInstances(X):
        return np.log(Metafeatures.numberOfInstances(X))
    
    
    @staticmethod
    def numberOfFeatures(X):
        return float(X.shape[1])
    
    @staticmethod
    def logNumberOfFeatures(X):
        return np.log(Metafeatures.numberOfFeatures(X))
    
    
    # X should have only labels column
    @staticmethod
    def numberOfClasses(X):
        return len(np.unique(X))
    
    
    #currently this is used to check for mistaken input 
    # in the future maybe we can generalize for dataset with missing data?
       
    # insted of using np.isfinite, 'None' value is checked
    @staticmethod
    def isMissingValues(X):
        return np.count_nonzero(X == None) + np.count_nonzero(X == '') > 0
    
    @staticmethod
    def numberOfMissingValues(X):
        return np.count_nonzero(X == None) + np.count_nonzero(X == '')
    
    @staticmethod
    def missingValuesRatio(X):
        return Metafeatures.numberOfMissingValues(X) / X.size
    
    
    @staticmethod
    def sparsity(X):
        return (np.count_nonzero(X == '') + np.count_nonzero(X == 0)) / X.size
    
    # X only has numerical columns
    @staticmethod
    def sparsityOnNumericColumns(X):
        return np.count_nonzero(X == 0) / X.size
    
    
    # columns of datasets are already classified in json files.
#     @staticmethod
#     def numberOfNumericFeatures(X):
#         #checks if the item is a string
#         count = 0
#         for item in X[1]:
#             if isNumerical(item):
#                 count += 1
#         return count

#     @staticmethod
#     def numberOfCategoricalFeatures(X):
#         # checks if the item is a string
#         count = 0
#         for item in X[1]:
#             if isCategorical(item):
#                 count += 1
#         return count


#     #returns True is any data is missing   
#     @staticmethod
#     def ratioNumericalToNominal(X):
#         num_categorical = float(numberOfCategoricalFeatures(X))
#         num_numerical = float(numberOfNumericFeatures(X))
#         if num_categorical == 0.0:
#             return 0.
#         else: 
#             return num_numerical / num_categorical
    
#     @staticmethod
#     def ratioNominalToNumerical(X):
#         num_categorical = float(numberOfCategoricalFeatures(X))
#         num_numerical = float(numberOfNumericFeatures(X))
#         if num_numerical == 0.0:
#             return 0.
#         else: 
#             return num_categorical/num_numerical

        
    @staticmethod
    def datasetRatio(X):
        return float(Metafeatures.numberOfFeatures(X)) /\
            float(Metafeatures.numberOfInstances(X))
    
    @staticmethod
    def logDatasetRatio(X):
        return np.log(Metafeatures.datasetRatio(X))
    
    
    # X only has numerical columns
    @staticmethod
    def minSkewness(X):
        skewness = scStat.skew(X)
        
        # only finite values are accepted
        skewness = skewness[np.isfinite(skewness)]
        
        return min(skewness)
    
    # X only has numerical columns
    @staticmethod
    def maxSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return max(skewness)
    
    # X only has numerical columns
    @staticmethod
    def medianSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.median(skewness)
    
    # X only has numerical columns
    @staticmethod
    def meanSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.mean(skewness)
    
    # X only has numerical columns
    @staticmethod
    def firstQuartileSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.percentile(skewness, 25)
    
    # X only has numerical columns
    @staticmethod
    def thirdQuartileSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.percentile(skewness, 75)
    
    
    # X only has numerical columns
    @staticmethod
    def minKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return min(kurtosis)
    
    # X only has numerical columns
    @staticmethod
    def maxKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return max(kurtosis)
    
    # X only has numerical columns
    @staticmethod
    def medianKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.median(kurtosis)
    
    # X only has numerical columns
    @staticmethod
    def meanKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.mean(kurtosis)
    
    # X only has numerical columns
    @staticmethod
    def firstQuartileKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.percentile(kurtosis, 25)
    
    # X only has numerical columns
    @staticmethod
    def thirdQuartileKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.percentile(kurtosis, 75)
    
    
    
#     @staticmethod
#     def Kurtosisses(X):
#         if isMissingFeatures(X):
#             print('Error in calculating Kurtosis, data is missing values')
#             return None
    
#         kurts = []
#         for i in range(X.shape[1]):
#             if not isCategorical(i):
#                 kurts.append(scipy.stats.kurtosis(X[:, i]))
#         return kurts

        
#     @staticmethod
#     def Skewnesses(X):
#         if isMissingFeatures(X):
#             print('Error in calculating Skewness, data is missing values')
#             return None
        
#         skews = []
#         for i in range(X.shape[1]):
#             if not isCategorical(i):
#                 skews.append(scipy.stats.skew(X[:, i]))
#         return skews
    
    # todo we may want to calculate some more nuanced metrics for kurtosis and skewness in the future.

    # todo not sure if landmarking can be done without labels, to look into and maybe implement later
    
    
    # X only has numerical columns
    @staticmethod
    def minCorrelation(X):
        corr = np.corrcoef(X.T)
        corr = corr[np.isfinite(corr)]
        return np.min(corr)
    
    # X only has numerical columns
    @staticmethod
    def maxCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        corr = corr[np.isfinite(corr)]
        return np.max(corr)
    
    # X only have numerical columns
    @staticmethod
    def medianCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.median(corr)
    
    # X only has numerical columns
    @staticmethod
    def meanCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.mean(corr)
    
    # X only has numerical columns
    @staticmethod
    def firstQuartileCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.percentile(corr, 25)
    
    # X only has numerical columns
    @staticmethod
    def thirdQuartileCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.percentile(corr, 75)
    
    
    # X only has numerical columns
    @staticmethod
    def minCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.min(cov)
    
    # X only has numerical columns
    @staticmethod
    def maxCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.max(cov)
    
    # X only has numerical columns
    @staticmethod
    def medianCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.median(cov)
    
    # X only has numerical columns
    @staticmethod
    def meanCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.mean(cov)
    
    # X only has numerical columns
    @staticmethod
    def firstQuartileCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.percentile(cov, 25)
    
    # X only has numerical columns
    @staticmethod
    def thirdQuartileCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.percentile(cov, 75)
    
    
#     @staticmethod
#     def PCA(X):
#         pca = sklearn.decomposition.PCA(copy=True)
#         rs = np.random.RandomState(42)
#         indices = np.arange(X.shape[0])
        
#         # replace missing values using the mean along each column
#         imp_mean = SimpleImputer(strategy='mean')
#         X_transformed = imp_mean.fit_transform(X)
        
#         for i in range(10):
#             try:
#                 rs.shuffle(indices)
#                 pca.fit(X_transformed[indices])
#                 return pca
#             except LinAlgError as e:
#                 pass
#         print("Failed to compute PCA")
#         #self.logger.warning("Failed to compute a Principle Component Analysis")
#         return None

#     @staticmethod
#     def PCAFractionOfComponentsFor95PercentVariance(X):
#         pca_ = PCA(X)
#         if pca_ is None:
#             return np.NaN
#         sum_ = 0.
#         idx = 0
#         while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
#             sum_ += pca_.explained_variance_ratio_[idx]
#             idx += 1
#         return float(idx)/float(X.shape[1])
    
#     @staticmethod
#     def PCAKurtosisFirstPC(X):
#         pca_ = PCA(X)
#         if pca_ is None:
#             return np.NaN
#         components = pca_.components_
#         pca_.components_ = components[:1]
#         transformed = pca_.transform(X)
#         pca_.components_ = components

#         kurtosis = scipy.stats.kurtosis(transformed)
#         return kurtosis[0]
    
#     @staticmethod
#     def PCASkewnessFirstPC(X):
#         pca_ = helper_functions.get_value("PCA")
#         if pca_ is None:
#             return np.NaN
#         components = pca_.components_
#         pca_.components_ = components[:1]
#         transformed = pca_.transform(X)
#         pca_.components_ = components

#         skewness = scipy.stats.skew(transformed)
#         return skewness[0]

    
    # X only has numerical columns
    @staticmethod
    def PCAFractionOfComponentsFor95PercentVariance(X):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        
        # replace missing values using the mean along each column
        imp_mean = SimpleImputer(strategy='mean')
        X_transformed = imp_mean.fit_transform(X)
        
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X_transformed[indices])
                sum_ = 0.
                idx = 0
                while sum_ < 0.95 and idx < len(pca.explained_variance_ratio_):
                    sum_ += pca.explained_variance_ratio_[idx]
                    idx += 1
                return float(idx)/float(X.shape[1])
            
            except LinAlgError as e:
                pass
            
        print("Failed to compute PCA")
        #self.logger.warning("Failed to compute a Principle Component Analysis")
        return np.nan
    
    
    @staticmethod
    def PCAKurtosisFirstPC(X):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        
        imp_mean = SimpleImputer(strategy='mean')
        X_transformed = imp_mean.fit_transform(X)
        
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X_transformed[indices])
                components = pca.components_
                pca.components_ = components[:1]
                transformed = pca.transform(X_transformed)
                pca.components_ = components
                kurtosis = scStat.kurtosis(transformed)
                return kurtosis[0]

            except LinAlgError as e:
                pass
            
        print("Failed to compute PCA")
        #self.logger.warning("Failed to compute a Principle Component Analysis")
        return np.nan
    
    
    @staticmethod
    def PCASkewnessFirstPC(X):
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        
        imp_mean = SimpleImputer(strategy='mean')
        X_transformed = imp_mean.fit_transform(X)
        
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X_transformed[indices])
                components = pca.components_
                pca.components_ = components[:1]
                transformed = pca.transform(X_transformed)
                pca.components_ = components
                skewness = scStat.skew(transformed)
                return skewness[0]

            except LinAlgError as e:
                pass
            
        print("Failed to compute PCA")
        #self.logger.warning("Failed to compute a Principle Component Analysis")
        return np.nan
    
    
    # X only has label columns
    @staticmethod
    def entropyOfClasses(X):
        X1 = X[X != None]
        X1 = X1[X1 != '']
        freq_dict = Counter(X1)
        probs = np.array([value / len(X1) for value in freq_dict.values()])
        return np.sum(-np.log2(probs) * probs)
    
    
    # X only has categorical columns
    @staticmethod
    def minEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.min(entropies) / np.log2(Metafeatures.numberOfInstances(X))
    
    # X only has categorical columns
    @staticmethod
    def maxEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.max(entropies) / np.log2(Metafeatures.numberOfInstances(X))
    
    # X only has categorical columns
    @staticmethod
    def medianEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.median(entropies) / np.log2(Metafeatures.numberOfInstances(X))
    
    # X only has categorical columns
    @staticmethod
    def meanEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.mean(entropies) / np.log2(Metafeatures.numberOfInstances(X))
    
    # X only has categorical columns
    @staticmethod
    def firstQuartileEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.percentile(entropies, 25) / np.log2(Metafeatures.numberOfInstances(X))
    
    # X only has categorical columns
    @staticmethod
    def thirdQuartileEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.percentile(entropies, 75) / np.log2(Metafeatures.numberOfInstances(X))
    
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def minMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
             for sublist in (X.T)[1:]]
        
        freq_dict_class = Counter(X[:, 0])
        prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
        prob_dicts = []
        for sublist in X.T[1:]:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            prob_dicts.append(prob_dict)
        
        mutInf = []
        i = 0
        for sublist in X_concat:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
                     in prob_dict.items()]
            mutInf.append(np.sum(probs))
            i += 1
            
        return np.min(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def maxMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
             for sublist in (X.T)[1:]]
        
        freq_dict_class = Counter(X[:, 0])
        prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
        prob_dicts = []
        for sublist in X.T[1:]:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            prob_dicts.append(prob_dict)
        
        mutInf = []
        i = 0
        for sublist in X_concat:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
                     in prob_dict.items()]
            mutInf.append(np.sum(probs))
            i += 1
            
        return np.max(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def medianMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
             for sublist in (X.T)[1:]]
        
        freq_dict_class = Counter(X[:, 0])
        prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
        prob_dicts = []
        for sublist in X.T[1:]:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            prob_dicts.append(prob_dict)
        
        mutInf = []
        i = 0
        for sublist in X_concat:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
                     in prob_dict.items()]
            mutInf.append(np.sum(probs))
            i += 1
            
        return np.median(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def meanMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
             for sublist in (X.T)[1:]]
        
        freq_dict_class = Counter(X[:, 0])
        prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
        prob_dicts = []
        for sublist in X.T[1:]:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            prob_dicts.append(prob_dict)
        
        mutInf = []
        i = 0
        for sublist in X_concat:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
                     in prob_dict.items()]
            mutInf.append(np.sum(probs))
            i += 1
            
        return np.mean(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def firstQuartileMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
             for sublist in (X.T)[1:]]
        
        freq_dict_class = Counter(X[:, 0])
        prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
        prob_dicts = []
        for sublist in X.T[1:]:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            prob_dicts.append(prob_dict)
        
        mutInf = []
        i = 0
        for sublist in X_concat:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
                     in prob_dict.items()]
            mutInf.append(np.sum(probs))
            i += 1
            
        return np.percentile(mutInf, 25)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def thirdQuartileMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
             for sublist in (X.T)[1:]]
        
        freq_dict_class = Counter(X[:, 0])
        prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
        prob_dicts = []
        for sublist in X.T[1:]:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            prob_dicts.append(prob_dict)
        
        mutInf = []
        i = 0
        for sublist in X_concat:
            freq_dict = Counter(sublist)
            prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
            probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
                     in prob_dict.items()]
            mutInf.append(np.sum(probs))
            i += 1
            
        return np.percentile(mutInf, 75)
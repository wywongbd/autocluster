import numpy as np
import scipy.stats as scStat
import sklearn.decomposition
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from collections import Counter


# all functions are based off of the implementation in autosklearn
# this class contains the static function we use to calculate metafeatures

########################
# type of X is ndarray #               
# X has no headers     #
########################

class GeneralMetafeature(object):
    # dataset contains the whole columns
    splitted_data_num = 1
    data_type = [['y_col', 'categorical_cols', 'numeric_cols', 'ordinal_cols']]
        
    @staticmethod
    def numberOfInstances(X):
        return float(X.shape[0])
        
    @staticmethod
    def logNumberOfInstances(X):
        return np.log(GeneralMetafeature.numberOfInstances(X))
        
    #currently this is used to check for mistaken input 
    # in the future maybe we can generalize for dataset with missing data?

    @staticmethod
    def numberOfMissingValues(X):
        return np.count_nonzero(X == None) + np.count_nonzero(X == '')

    @staticmethod
    def missingValuesRatio(X):
        return GeneralMetafeature.numberOfMissingValues(X) / X.size

    @staticmethod
    def sparsity(X):
        return (np.count_nonzero(X == '') + np.count_nonzero(X == 0)) / X.size



class GeneralMetafeatureWithoutLabels(object):
    # dataset doesn't contain the label column
    splitted_data_num = 1
    data_type = [['categorical_cols', 'numeric_cols', 'ordinal_cols']]
        
    @staticmethod
    def numberOfFeatures(X):
        return float(X.shape[1])

    @staticmethod
    def logNumberOfFeatures(X):
        return np.log(GeneralMetafeatureWithoutLabels.numberOfFeatures(X))
        
    @staticmethod
    def datasetRatio(X):
        return float(GeneralMetafeatureWithoutLabels.numberOfFeatures(X)) /\
            float(GeneralMetafeature.numberOfInstances(X))

    @staticmethod
    def logDatasetRatio(X):
        return np.log(GeneralMetafeatureWithoutLabels.datasetRatio(X))



class LabelsMetafeatures(object):
    # dataset contains the label column only
    splitted_data_num = 1
    data_type = [['y_col']]
        
    @staticmethod
    def numberOfClasses(X):
        return len(np.unique(X))
        
    @staticmethod
    def entropyOfClasses(X):
        X1 = X[X != None]
        X1 = X1[X1 != '']
        freq_dict = Counter(X1)
        probs = np.array([value / len(X1) for value in freq_dict.values()])
        return np.sum(-np.log2(probs) * probs)

    
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


class NumericMetafeature(object):
    # dataset contains numeric columns only
    splitted_data_num = 1
    data_type = [['numeric_cols']]

    @staticmethod
    def sparsityOnNumericColumns(X):
        return np.count_nonzero(X == 0) / X.size

    @staticmethod
    def minSkewness(X):
        skewness = scStat.skew(X)
        # only finite values are accepted
        skewness = skewness[np.isfinite(skewness)]
        return min(skewness)

    @staticmethod
    def maxSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return max(skewness)

    @staticmethod
    def medianSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.median(skewness)

    @staticmethod
    def meanSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.mean(skewness)

    @staticmethod
    def firstQuartileSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.percentile(skewness, 25)

    @staticmethod
    def thirdQuartileSkewness(X):
        skewness = scStat.skew(X)
        skewness = skewness[np.isfinite(skewness)]
        return np.percentile(skewness, 75)


    @staticmethod
    def minKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return min(kurtosis)

    @staticmethod
    def maxKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return max(kurtosis)

    @staticmethod
    def medianKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.median(kurtosis)

    @staticmethod
    def meanKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.mean(kurtosis)

    @staticmethod
    def firstQuartileKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.percentile(kurtosis, 25)

    @staticmethod
    def thirdQuartileKurtosis(X):
        kurtosis = scStat.kurtosis(X)
        kurtosis = kurtosis[np.isfinite(kurtosis)]
        return np.percentile(kurtosis, 75)


    @staticmethod
    def minCorrelation(X):
        corr = np.corrcoef(X.T)
        corr = corr[np.isfinite(corr)]
        return np.min(corr)

    @staticmethod
    def maxCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        corr = corr[np.isfinite(corr)]
        return np.max(corr)

    @staticmethod
    def medianCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.median(corr)

    @staticmethod
    def meanCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.mean(corr)

    @staticmethod
    def firstQuartileCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.percentile(corr, 25)

    @staticmethod
    def thirdQuartileCorrelation(X):
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, np.nan)
        corr = corr.flatten()
        corr = corr[np.isfinite(corr)]
        return np.percentile(corr, 75)


    @staticmethod
    def minCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.min(cov)

    @staticmethod
    def maxCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.max(cov)

    @staticmethod
    def medianCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.median(cov)

    @staticmethod
    def meanCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.mean(cov)

    @staticmethod
    def firstQuartileCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.percentile(cov, 25)

    @staticmethod
    def thirdQuartileCovariance(X):
        cov = np.cov(X.T)
        cov = cov[np.isfinite(cov)]
        return np.percentile(cov, 75)


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



class CategoricalMetafeature(object):
    # dataset contains categorical columns only
    splitted_data_num = 1
    data_type = [['categorical_cols', 'ordinal_cols']]
    
    @staticmethod
    def minEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.min(entropies) / np.log2(GeneralMetafeature.numberOfInstances(X))
    
    @staticmethod
    def maxEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.max(entropies) / np.log2(GeneralMetafeature.numberOfInstances(X))
    
    @staticmethod
    def medianEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.median(entropies) / np.log2(GeneralMetafeature.numberOfInstances(X))
    
    @staticmethod
    def meanEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.mean(entropies) / np.log2(GeneralMetafeature.numberOfInstances(X))
    
    @staticmethod
    def firstQuartileEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.percentile(entropies, 25) / np.log2(GeneralMetafeature.numberOfInstances(X))
    
    @staticmethod
    def thirdQuartileEntropy(X):
        entropies = []
        for sublist in X.T:
            sublist1 = sublist[sublist != None]
            sublist1 = sublist1[sublist1 != '']
            freq_dict = Counter(sublist1)
            probs = np.array([value / len(sublist1) for value in freq_dict.values()])
            entropies.append(np.sum(-np.log2(probs) * probs))
        return np.percentile(entropies, 75) / np.log2(GeneralMetafeature.numberOfInstances(X))
    
    
    
class CategoricalMetafeatureWithLabels(object):
    # dataset contains categorical columns and the label column
    splitted_data_num = 2
    data_type = [['y_col'], ['categorical_cols', 'ordinal_cols']]
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def minMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        # code without library
#         X_concat = [[tuple(value) for value in np.concatenate((X[:, :1], np.reshape(sublist, (-1, 1))), axis=1)]\
#              for sublist in (X.T)[1:]]
        
#         freq_dict_class = Counter(X[:, 0])
#         prob_dict_class = dict([[key, value / len(sublist)] for (key, value) in freq_dict_class.items()])
#         prob_dicts = []
#         for sublist in X.T[1:]:
#             freq_dict = Counter(sublist)
#             prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
#             prob_dicts.append(prob_dict)
        
#         mutInf = []
#         i = 0
#         for sublist in X_concat:
#             freq_dict = Counter(sublist)
#             prob_dict = dict([[key, value / len(sublist)] for (key, value) in freq_dict.items()])
#             probs = [value * np.log(value / prob_dict_class[key[0]] / (prob_dicts[i])[key[1]]) for (key, value)\
#                      in prob_dict.items()]
#             mutInf.append(np.sum(probs))
#             i += 1

        mutInf = []
        class_col1 = X[:, 0]
        for sublist in X.T[1:]:
            mutInf.append(mutual_info_score(labels_true = class_col1, labels_pred = sublist))
            
        return np.min(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def maxMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        mutInf = []
        class_col1 = X[:, 0]
        for sublist in X.T[1:]:
            mutInf.append(mutual_info_score(labels_true = class_col1, labels_pred = sublist))
            
        return np.max(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def medianMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        mutInf = []
        class_col1 = X[:, 0]
        for sublist in X.T[1:]:
            mutInf.append(mutual_info_score(labels_true = class_col1, labels_pred = sublist))
            
        return np.median(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def meanMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        mutInf = []
        class_col1 = X[:, 0]
        for sublist in X.T[1:]:
            mutInf.append(mutual_info_score(labels_true = class_col1, labels_pred = sublist))
            
        return np.mean(mutInf)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def firstQuartileMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        mutInf = []
        class_col1 = X[:, 0]
        for sublist in X.T[1:]:
            mutInf.append(mutual_info_score(labels_true = class_col1, labels_pred = sublist))
            
        return np.percentile(mutInf, 25)
    
    # class_col has the label column only, cat_cols has categorical columns only
    @staticmethod
    def thirdQuartileMutualInformation(class_col, cat_cols):
        X = np.concatenate((class_col, cat_cols), axis=1)
        X = X[~(X == '').any(axis=1)]
        X = X[~(X == None).any(axis=1)]
        
        mutInf = []
        class_col1 = X[:, 0]
        for sublist in X.T[1:]:
            mutInf.append(mutual_info_score(labels_true = class_col1, labels_pred = sublist))
            
        return np.percentile(mutInf, 75)
    
    
#########################################################################################################

    
class MetafeatureMapper(object):
    feature_type = {
        f: GeneralMetafeature for f in list(GeneralMetafeature.__dict__) if '_' not in f
    }
    feature_type.update({f: GeneralMetafeatureWithoutLabels for f in list(GeneralMetafeatureWithoutLabels.__dict__) if '_' not in f})
    feature_type.update({f: NumericMetafeature for f in list(NumericMetafeature.__dict__) if '_' not in f})
    feature_type.update({f: CategoricalMetafeature for f in list(CategoricalMetafeature.__dict__) if '_' not in f})
    feature_type.update({f: CategoricalMetafeatureWithLabels for f in list(CategoricalMetafeatureWithLabels.__dict__) if '_' not in f})
    
    feature_function = {
        f_name: f_obj for f_name, f_obj in (GeneralMetafeature.__dict__).items() if '_' not in f_name
    }
    feature_function.update({f_name: f_obj for f_name, f_obj in (GeneralMetafeatureWithoutLabels.__dict__).items() if '_' not in f_name})
    feature_function.update({f_name: f_obj for f_name, f_obj in (NumericMetafeature.__dict__).items() if '_' not in f_name})
    feature_function.update({f_name: f_obj for f_name, f_obj in (CategoricalMetafeature.__dict__).items() if '_' not in f_name})
    feature_function.update({f_name: f_obj for f_name, f_obj in (CategoricalMetafeatureWithLabels.__dict__).items() if '_' not in f_name})
    
    @staticmethod
    def getClass(string):
        return MetafeatureMapper.feature_type.get(string, None)
    
    @staticmethod
    def getMetafeatureFunction(string):
        return MetafeatureMapper.feature_function.get(string, None)
    
    @staticmethod
    def getAllMetafeatures():
        return list(MetafeatureMapper.feature_type.keys())
    
    @staticmethod
    def getGeneralMetafeatures():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is GeneralMetafeature]
    
    @staticmethod
    def getGeneralMetafeaturesWithoutLabels():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is GeneralMetafeatureWithoutLabels]
    
    @staticmethod
    def getNumericMetafeatures():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is NumericMetafeature]
    
    @staticmethod
    def getCategoricalMetafeatures():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is CategoricalMetafeature]
    
    @staticmethod
    def getCategoricalMetafeaturesWithLabels():
        return [key for key, value in MetafeatureMapper.feature_type.items() if value is CategoricalMetafeatureWithLabels]
    
    
    
def calculate_metafeatures(raw_dataset, file_dict, metafeature_ls = []):
    if len(metafeature_ls) == 0:
        metafeature_ls = MetafeatureMapper.getAllMetafeatures()
        
    values = []
    
    for feature_str in metafeature_ls:
        feature_class = MetafeatureMapper.getClass(feature_str)
        datasets = []
        app_data_type = True
        
        for i in range(feature_class.splitted_data_num):
            col_list = []
            
            for col in feature_class.data_type[i]:
                if file_dict[col]:
                    temp_data = raw_dataset[file_dict[col]].to_numpy()
                    if temp_data.ndim == 1:
                        temp_data = np.reshape(temp_data, (-1, 1))
                    col_list.append(temp_data)
                else:
                    values.append(None)
                    app_data_type = False
                    break
                    
            if app_data_type:
                datasets.append(np.concatenate(tuple(col_list), axis=1))
            else:
                break
            
        if app_data_type:
            values.append(MetafeatureMapper.getMetafeatureFunction(feature_str).__get__(object)(*datasets))
        
    return np.reshape(np.array(values), (-1, 1))
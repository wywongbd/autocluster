from sklearn import metrics
from collections import Counter
import numpy as np

class Evaluators(object):
    @staticmethod
    def silhouetteScore(X, y_pred):
        return (1 - metrics.silhouette_score(X, y_pred)) / 2
    
    @staticmethod
    def daviesBouldinScore(X, y_pred):
        return np.tanh(metrics.davies_bouldin_score(X, y_pred))
    
    @staticmethod
    def calinskiHarabaszScore(X, y_pred):
        return 1 - np.tanh(metrics.calinski_harabasz_score(X, y_pred))
    
    
class EvaluatorMapper(object):
    eval_function = {
        f_name: f_obj for f_name, f_obj in (Evaluators.__dict__).items() if '_' not in f_name
    }
    
    @staticmethod
    def getEvaluatorFunction(string):
        return EvaluatorMapper.eval_function.get(string, None)
    
    @staticmethod
    def getAllEvaluators():
        return list(EvaluatorMapper.eval_function.keys())
    
    @staticmethod
    def linearCombinationOfEvaluators(X, y_pred, evaluator_ls, weights, clustering_num, min_proportion, min_relative_proportion):
        # check if num of clusters is within the range
        if type(clustering_num) == int:
            if len(set(y_pred)) != clustering_num:
                return float('inf')
        elif type(clustering_num) == tuple:
            if len(clustering_num) == 1 and clustering_num[0] > 1:
                if len(set(y_pred)) != clustering_num[0]:
                    return float('inf')
            elif len(clustering_num) == 2 and clustering_num[0] <= clustering_num[1] and clustering_num[1] > 1:
                if len(set(y_pred)) < clustering_num[0] or len(set(y_pred)) == 1 or len(set(y_pred)) > clustering_num[1]:
                    return float('inf')

        if len(set(y_pred)) == 1 :
            return float('inf')
        
        # check if (minimun cluster size)/(sum of all cluster size) is over min_proportion
        freq_dict = Counter(y_pred)
        cluster_size_ls = list(freq_dict.values())
        min_cluster_size = np.min(cluster_size_ls)
        if min_cluster_size / len(y_pred) < min_proportion:
            return float('inf')
        
        # check if (minimun cluster size)/(max cluster size) is over min_relative_proportion
        if type(min_relative_proportion) == float or min_relative_proportion == 'default':
            if min_relative_proportion == 'default':
                min_relative_proportion_ = 5 * min_proportion
            else:
                min_relative_proportion_ = min_relative_proportion
            
            max_cluster_size = np.max(cluster_size_ls)
            if min_cluster_size / max_cluster_size < min_relative_proportion_:
                return float('inf')
    
        # evaluate linear combination of scores
        values = []

        for evaluator in evaluator_ls:
            values.append(EvaluatorMapper.getEvaluatorFunction(evaluator).__get__(object)(X, y_pred))

        weight_sum = np.sum(weights)
        if len(weights) != len(evaluator_ls) or weight_sum == 0:
            return np.mean(values)
        else:
            return np.sum(np.multiply(weights, values)) / weight_sum
    
    
    
def get_evaluator(evaluator_ls = ['silhouetteScore'], weights = [], clustering_num = None, min_proportion = .01,\
                  min_relative_proportion = 'default'):
    # evaluator_ls : 'silhouetteScore' or 'daviesBouldinScore' or 'calinskiHarabaszScore'
    # weights : coefficients of evaluators. no need to make total = 1, but should not make total = 0
    # clustering_num : integer or tuple. None is equal to (2, float('inf'))
    # min_proportion : (minimun cluster size)/(maximun cluster size)
    # min_relative_proportion : (minimun cluster size)/(max cluster size)
    #                           if min_relative_proportion=='default', then min_relative_proportion = 5 * min_proportion
    #                           if min_relative_proportion==None, then ignore min_relative_proportion
    
    return (lambda X, y_pred: EvaluatorMapper.linearCombinationOfEvaluators(X, y_pred, evaluator_ls = evaluator_ls,\
                            weights = weights, clustering_num = clustering_num, min_proportion = min_proportion,\
                            min_relative_proportion = min_relative_proportion))
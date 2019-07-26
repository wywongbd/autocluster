from sklearn import metrics
import numpy as np

class Evaluators(object):
    @staticmethod
    def silhouetteScore(X, y_pred):
        return (1 - metrics.silhouette_score(X, y_pred)) / 2
    
    @staticmethod
    def daviesBouldinScore(X, y_pred):
        return np.tanh(davies_bouldin_score(X, y_pred))
    
    @staticmethod
    def calinskiHarabaszScore(X, y_pred):
        return 1 - np.tanh(calinski_harabasz_score(X, y_pred))
    
    
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
    def linearCombinationOfEvaluators(X, y_pred, evaluator_ls, weights, clustering_num, min_proportion):
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
    
        values = []

        for evaluator in evaluator_ls:
            values.append(EvaluatorMapper.getEvaluatorFunction(evaluator).__get__(object)(X, y_pred))

        weight_sum = np.sum(weights)
        if len(weights) != len(evaluator_ls) or weight_sum == 0:
            return np.mean(values)
        else:
            return np.sum(np.multiply(weights * values)) / weight_sum
    
    
    
def get_evaluator(evaluator_ls = ['silhouetteScore'], weights = []):
    return (lambda X, y_pred: EvaluatorMapper.linearCombinationOfEvaluators(X, y_pred,\
                                evaluator_ls = evaluator_ls, weights = weights))
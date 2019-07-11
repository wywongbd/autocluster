from algorithms import algorithms
from utils.stringutils import StringUtils

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

class Mapper(object):
    d = {
        "KMeans": algorithms.KMeans,
        "DBSCAN": algorithms.DBSCAN,
        "MiniBatchKMeans": algorithms.MiniBatchKMeans,
        "AffinityPropagation": algorithms.AffinityPropagation,
        "MeanShift": algorithms.MeanShift,
        "SpectralClustering": algorithms.SpectralClustering,
        "AgglomerativeClustering": algorithms.AgglomerativeClustering,
        "OPTICS": algorithms.OPTICS,
        "Birch": algorithms.Birch,
        "GaussianMixture": algorithms.GaussianMixture
    }
    @staticmethod
    def getClass(string):
        return Mapper.d[string]
    
    @staticmethod
    def getAlgorithms():
        return list(Mapper.d.keys())

def build_config_space(algorithms_ls=["KMeans", "DBSCAN"]):
    cs = ConfigurationSpace()
    algorithm_choice = CategoricalHyperparameter("algorithm_choice", 
                                                 algorithms_ls, 
                                                 default_value=algorithms_ls[0])
    cs.add_hyperparameters([algorithm_choice])
    
    for string in algorithms_ls:
        algorithm = Mapper.getClass(string)
        
        # encode parameter names
        encoded_params = []
        for param in algorithm.params:
            encoded_string = StringUtils.encode_parameter(param.name, algorithm.name)
            param.name = encoded_string
        
        # add encoded paramters to configuration space
        cs.add_hyperparameters(algorithm.params)
        
        # define dependency
        for param in algorithm.params:
            cs.add_condition(InCondition(child=param, parent=algorithm_choice, values=[string]))
        
        # add forbidden clauses
        for condition in algorithm.forbidden_clauses:
            cs.add_forbidden_clause(condition)
    
    return cs
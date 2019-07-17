import itertools

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
        "GaussianMixture": algorithms.GaussianMixture,
        "TSNE": algorithms.TSNE,
		"PCA": algorithms.PCA,
        "IncrementalPCA": algorithms.IncrementalPCA,
        "LatentDirichletAllocation": algorithms.LatentDirichletAllocation
    }
    @staticmethod
    def getClass(string):
        return Mapper.d.get(string, None)
    
    @staticmethod
    def getAlgorithms():
        return list(Mapper.d.keys())

def build_config_space(clustering_ls=["KMeans", "DBSCAN"], dim_reduction_ls=[]):
    cs = ConfigurationSpace()
    
    if len(clustering_ls) > 0:
        clustering_choice = CategoricalHyperparameter("clustering_choice", 
                                                      clustering_ls, 
                                                      default_value=clustering_ls[0])
        cs.add_hyperparameters([clustering_choice])
    
    if len(dim_reduction_ls) > 0:
        dim_reduction_choice = CategoricalHyperparameter("dim_reduction_choice", 
                                                         dim_reduction_ls,
                                                         default_value=dim_reduction_ls[0])
        cs.add_hyperparameters([dim_reduction_choice])    
    
    for idx, string in enumerate(itertools.chain(clustering_ls, dim_reduction_ls)):
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
            cs.add_condition(InCondition(child=param, 
                                         parent=clustering_choice if idx < len(clustering_ls) \
                                         else dim_reduction_choice, 
                                         values=[string]))
        
        # add forbidden clauses
        for condition in algorithm.forbidden_clauses:
            cs.add_forbidden_clause(condition)
    
    return cs
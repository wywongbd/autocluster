import itertools

from .algorithms import algorithms
from .utils.stringutils import StringUtils

from smac.configspace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

class Mapper(object):
    d = {
        class_name: class_obj for class_name, class_obj in (algorithms.__dict__).items() 
        if '_' not in class_name
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


def build_config_obj(config_space, values_dict):
    unconditional_parameters = config_space.get_all_unconditional_hyperparameters()
    
    for hyperparam in unconditional_parameters:
        choice = values_dict.get(hyperparam, None)
        if choice is None:
            continue
            
        algorithm = Mapper.getClass(choice)
        for param in algorithm.params:
            param_name = StringUtils.encode_parameter(param.name, algorithm.name)
            if param_name not in values_dict:
                values_dict[param_name] = param.default_value
                
    return Configuration(configuration_space=config_space, values=values_dict)
                
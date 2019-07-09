from algorithms import algorithms
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

class Mapper(object):
    d = {
        "KMeans": algorithms.KMeans,
        "DBSCAN": algorithms.DBSCAN
    }
    @staticmethod
    def getClass(string):
        return Mapper.d[string]


def build_config_space(algorithms_ls=["KMeans", "DBSCAN"]):
    cs = ConfigurationSpace()
    algorithm_choice = CategoricalHyperparameter("algorithm_choice", 
                                                 algorithms_ls, 
                                                 default_value=algorithms_ls[0])
    cs.add_hyperparameters([algorithm_choice])
    
    for string in algorithms_ls:
        algorithm = Mapper.getClass(string) 
        cs.add_hyperparameters(algorithm.params)
        for param in algorithm.params:
            cs.add_condition(InCondition(child=param, parent=algorithm_choice, values=[string]))
    
    return cs
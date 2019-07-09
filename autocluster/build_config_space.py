from algorithms import algorithms
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

def build_config_space(algorithms_ls=[algorithms.KMeans, algorithms.DBSCAN]):
    cs = ConfigurationSpace()
    algorithm_choice = CategoricalHyperparameter("algorithm_choice", 
                                                 [alg.name for alg in algorithms_ls], 
                                                 default_value=algorithms_ls[0].name)
    cs.add_hyperparameters([algorithm_choice])
    
    for algorithm in algorithms_ls:
        print(algorithm)
        cs.add_hyperparameters(algorithm.params)
        for param in algorithm.params:
            cs.add_condition(InCondition(child=param, parent=algorithm_choice, values=[algorithm.name]))
    
    return cs
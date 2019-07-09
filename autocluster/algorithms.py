from sklearn import cluster 
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter

class ClusteringAlgorithm(type):
    # metaclass to ensure that static variables in the classes below are read-only
    @property
    def name(cls):
        return cls._name
    
    @property
    def model(cls):
        return cls._model
    
    @property
    def params(cls):
        return cls._params
    
    @property
    def params_names(cls):
        return cls._params_names


class dbscan(object, metaclass=ClusteringAlgorithm):
    # static variables
    _name = "dbscan"
    _model = cluster.DBSCAN
    _params = [
        UniformFloatHyperparameter("eps", 0.01, 10, default_value=2.0),
        UniformIntegerHyperparameter("min_samples", 5, 1000, default_value=100)
    ]
    _params_names = set([p.name for p in _params])
    
class kmeans(object, metaclass=ClusteringAlgorithm):
    # static variables
    _name = "kmeans"
    _model = cluster.KMeans
    _params = [
        UniformIntegerHyperparameter("n_clusters", 1, 20, default_value=10)
    ]
    _params_names = set([p.name for p in _params]) 
    
    


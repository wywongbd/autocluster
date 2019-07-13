from sklearn import cluster, mixture, manifold
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause

class algorithms(object):
    # this class is just to create an extra layer of namespace
    
    class Metaclass(type):
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
        
        @property
        def conditions(cls):
            return cls._conditions
        
        @property
        def forbidden_clauses(cls):
            return cls._forbidden_clauses

    class DBSCAN(object, metaclass=Metaclass):
        # static variables
        _name = "DBSCAN"
        _model = cluster.DBSCAN
        _params = [
            UniformFloatHyperparameter("eps", 0.01, 5, default_value=0.01),
            UniformIntegerHyperparameter("min_samples", 5, 100, default_value=5)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class KMeans(object, metaclass=Metaclass):
        # static variables
        _name = "KMeans"
        _model = cluster.KMeans
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 15, default_value=5)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class MiniBatchKMeans(object, metaclass=Metaclass):
        # static variables
        _name = "MiniBatchKMeans"
        _model = cluster.MiniBatchKMeans
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 15, default_value=10),
            UniformIntegerHyperparameter("batch_size", 10, 1000, default_value=100)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
    
    class AffinityPropagation(object, metaclass=Metaclass):
        # static variables
        _name = "AffinityPropagation"
        _model = cluster.AffinityPropagation
        _params = [
            UniformFloatHyperparameter("damping", 0.5, 1, default_value=0.5),
            
            # "affinity" was added
            CategoricalHyperparameter("affinity", ['euclidean'], default_value='euclidean')
            
            # 'precomputed' is excluded from "affinity"s possible values
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class MeanShift(object, metaclass=Metaclass):
        # static variables
        _name = "MeanShift"
        _model = cluster.MeanShift
        _params = [
            CategoricalHyperparameter("bin_seeding", [True, False], default_value=False)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class SpectralClustering(object, metaclass=Metaclass):
        # static variables
        _name = "SpectralClustering"
        _model = cluster.SpectralClustering
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 20, default_value=10),
            
            # None and 'lobpcg' were excluded from eigne_solver's list of possible values
            CategoricalHyperparameter("eigen_solver", ['arpack'], default_value='arpack'),
            
            # Values 'poly', 'sigmoid', 'laplacian', 'chi2' were included,
            # 'precomputed' is excluded because it requires distance matrix input
            # 'chi2' is excluded due to "ValueError: X contains negative values.""
            CategoricalHyperparameter("affinity", ['nearest_neighbors', 'poly', 'sigmoid',\
                                                   'laplacian', 'rbf'], default_value='rbf'),
            
            # "assign_labels" was added
            CategoricalHyperparameter("assign_labels", ['kmeans','discretize'], default_value='kmeans')
            # -----------------------------------------------------------------
            # TODO:
            # -----------------------------------------------------------------
            # error was found when 'amg' was passed into 'eigen_solver'
            # ValueError: The eigen_solver was set to 'amg', but pyamg is not available.
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class AgglomerativeClustering(object, metaclass=Metaclass):
        # static variables
        _name = "AgglomerativeClustering"
        _model = cluster.AgglomerativeClustering
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 20, default_value=10),
            CategoricalHyperparameter("linkage", 
                                      ['ward', 'complete', 'average', 'single'], 
                                      default_value='complete'),
            CategoricalHyperparameter("affinity", 
                                      ['euclidean', 'cityblock', 
                                       'l2', 'l1', 'manhattan', 'cosine'],
                                      default_value='euclidean')
            #'ward' has been included now
            # 'precomputed' has been excluded from "affinity" s possible values because it requires 
            # a precomputed distance matrix as input from user
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = [
            ForbiddenAndConjunction(ForbiddenEqualsClause(_params[1], "ward"), 
                                    ForbiddenInClause(_params[2], ['cosine', 'cityblock', 
                                                                   'l2', 'l1', 'manhattan']))
        ]
        
    class OPTICS(object, metaclass=Metaclass):
        # static variables
        _name = "OPTICS"
        _model = cluster.OPTICS
        _params = [
            UniformIntegerHyperparameter("min_samples", 5, 1000, default_value=100),
            
            # "max_eps" may not be useful
            #UniformFloatHyperparameter("max_eps", 0.01, 10, default_value=2.0),
            
            CategoricalHyperparameter("metric", ['minkowski', 'euclidean', 
                                                 'manhattan', 'l1', 'l2', 'cosine'], default_value='minkowski'),
            CategoricalHyperparameter("cluster_method", ['xi', 'dbscan'], default_value='xi')
            
            # -----------------------------------------------------------------
            # TODO:
            # -----------------------------------------------------------------
            # some metrics, like the following, are only for boolean arrays, and will lead to infinite recursion when passed in with non-boolean data
            # 'russellrao', 'sokalmichener', 'dice', 'rogerstanimoto'
            # due to this reason, the 'metric' 
            # orginal entire list of metrics:
            # ['euclidean', 'l1', 'l2', 'manhattan',\
            #   'cosine', 'cityblock', 'braycurtis',\
            #   'canberra', 'chebyshev', 'correlation',\
            #   'hamming', 'jaccard', 'kulsinski',\
            #   'mahalanobis', 'minkowski',\
            #   'seuclidean', 'russellrao', 'sokalmichener', 'dice', 'rogerstanimoto', \
            #   'sokalsneath', 'sqeuclidean', 'yule'],\
            #
            # perhaps 'metric' should be an input from user, we don't need to optimize it at all
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class Birch(object, metaclass=Metaclass):
        # static variables
        _name = "Birch"
        _model = cluster.Birch
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 20, default_value=5),
            
            # "branching_factor" was added
            UniformIntegerHyperparameter("branching_factor", 10, 1000, default_value=50)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class GaussianMixture(object, metaclass=Metaclass):
        # static variables
        _name = "GaussianMixture"
        _model = mixture.GaussianMixture
        _params = [
            UniformIntegerHyperparameter("n_components", 1, 20, default_value=5),
            CategoricalHyperparameter("covariance_type", ['full', 'tied', 'diag', 'spherical'], default_value='full'),
            CategoricalHyperparameter("init_params", ['kmeans', 'random'], default_value='kmeans'),
            CategoricalHyperparameter("warm_start", [True, False], default_value=False)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class TSNE(object, metaclass=Metaclass):
        # static variables
        _name = "TSNE"
        _model = manifold.TSNE
        _params = [
            UniformIntegerHyperparameter("n_components", 2, 3, default_value=2),
            UniformFloatHyperparameter("perplexity", 5, 50, default_value=30)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    
from dataset import Dataset

class AutoCluster(object):
    def __init__(self):
        self._algorithm = None
        self._dataset = None
    
    def fit(self, X):
        # create dataset object
        self._dataset = Dataset(X)
            
    
    def plot_clusters(self):
        pass
    
    def predict(self, X):
        return self.algorithm.predict(X) if self.algorithm else None
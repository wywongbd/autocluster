from dataset import Dataset

class AutoCluster(object):
    def __init__(self):
        self._dataset = None
    
    def fit(self, X):
        # create dataset object
        self._dataset = Dataset(X)
        pass
    
    def plot_clusters(self):
        pass
    
    def predict(self, X):
        return self.algorithm.predict(X) if self.algorithm else None
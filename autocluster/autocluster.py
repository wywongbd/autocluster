from dataset import Dataset

class AutoCluster(object):
    def __init__(self):
        self.algorithm = None
    
    def fit(self, X, standardize=True):
            
    
    def plot_clusters(self):
        pass
    
    def predict(self, X):
        return self.algorithm.predict(X) if self.algorithm else None
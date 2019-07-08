from sklearn.preprocessing import StandardScaler, RobustScaler

class Dataset(object):
    def __init__(self, X):
        self._X = None
        self._standard_scaler = None
        self._robust_scaler = None
        
        # invoke setter
        self.data = X
    
    @property
    def data(self):
        return self._X
    
    @data.setter
    def data(self, new_data):
        self._X = new_data
        self._standard_scaler = StandardScaler()
        self._robust_scaler = RobustScaler()
        self._standard_scaler.fit(self._X)
        self._robust_scaler.fit(self._X)
        
    @property
    def standard_scaler(self):
        return self._standard_scaler
    
    @property
    def robust_scaler(self):
        return self._robust_scaler
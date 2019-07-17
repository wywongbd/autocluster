import numpy as np

class Metafeatures(object):
    @staticmethod
    def numberOfInstances(X):
        return len(X)

    @staticmethod
    def numberOfFeatures(X):
        return X.shape[1]

    @staticmethod
    def numberOfNumericFeatures(X):
        # checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is not str:
                count += 1
        return count

    @staticmethod
    def numberOfCategoricalFeatures(X):
        # checks if the item is a string
        count = 0
        for item in X[1]:
            if type(item) is str:
                count += 1
        return count

    # @staticmethod
    # def numberOfInstances(X):
    #     # checks if the item is a string
    #     count = 0
    #     for item in X[1]:
    #         if type(item) is not str:
    #             count += 1
    #     return count

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class ClusterUtils(object):
    @staticmethod
    def generate_sample_data(n_samples=1500):
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                              noise=.05)
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
        no_structure = np.random.rand(n_samples, 2), None

        # Anisotropicly distributed data
        random_state = 170
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(n_samples=n_samples,
                                     cluster_std=[1.0, 2.5, 0.5],
                                     random_state=random_state)
        
        # return as a huge tuple
        return varied, blobs, no_structure, noisy_circles, noisy_moons, aniso
    
    @staticmethod
    def visualize_sample_data(points):
        # to use this method, just do something like: 
        # visualize_sample_data(varied[0])
        # visualize_sample_data(blobs[0])
        # visualize_sample_data(no_structure[0])
        # visualize_sample_data(noisy_circles[0])
        # visualize_sample_data(noisy_moons[0])
        # visualize_sample_data(aniso[0])
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()
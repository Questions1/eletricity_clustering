import numpy as np
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 1) + np.array([20])

# generate zero centered stretched Gaussian data
C = np.array([[1]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 1), C)

another_gaussian = np.dot(np.random.randn(n_samples, 1), C.T) + np.array([10])

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian, another_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(np.array(grey).reshape(-1, 1))


clf.means_
clf.covariances_
predicted = clf.predict(np.array([0, 10, 20]).reshape(-1, 1))

predicted = clf.predict(np.array(grey).reshape(-1, 1))
















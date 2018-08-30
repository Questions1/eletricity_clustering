import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import preprocessing


X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5,
                           cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
X = data_final['Ia'].values.reshape(-1, 1)
y_pred = SpectralClustering().fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

for index, gamma in enumerate((0.01, 0.1, 1, 10)):
    for index, k in enumerate((2, 3, 4, 5)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=",
              k, "score:", metrics.calinski_harabaz_score(X, y_pred))

y_pred = SpectralClustering(gamma=0.1).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))


X = np.array([1., 1, 1, 1, 30, 30, 30, 30, 50, 50, 50, 50]*1).reshape(-1, 1)
X = preprocessing.scale(X)

km = KMeans(n_clusters=2).fit(X)
metrics.calinski_harabaz_score(X, km.labels_)

km = KMeans(n_clusters=3).fit(X)
metrics.calinski_harabaz_score(X, km.labels_)

X = np.array([1., 1, 1, 1, 30, 30, 30, 30, 50, 50, 50, 50]*1).reshape(-1, 1)
X = preprocessing.scale(X)
for index, k in enumerate((2, 3, 4, 5, 6)):
    km = KMeans(n_clusters=k).fit(X)
    # y_pred = km.predict(X)
    print("Calinski-Harabasz Score with n_clusters=%s score:%s"
          % (k, metrics.calinski_harabaz_score(X, km.labels_)))

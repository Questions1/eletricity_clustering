import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_distance(data):
    distance = []
    k = []
    for n_clusters in range(1, 10):
        cls = KMeans(n_clusters).fit(data)

        distance_sum = 0
        for i in range(n_clusters):
            group = (cls.labels_ == i)
            members = data[group]
            distance_sum += np.var(members - cls.cluster_centers_[i]).values[0]
        distance.append(distance_sum)
        k.append(n_clusters)
    plt.scatter(k, distance)
    plt.plot(k, distance)
    plt.xlabel("k")
    plt.ylabel("distance")
    plt.show()

    return distance, k


def get_elbow(distance):
    a_1 = pd.Series(distance[1:])
    a_2 = pd.Series(distance[:-1])

    tmp = a_2 / a_1
    return tmp.values.argmax() + 2


if __name__ == '__main__':
    a = pd.DataFrame(np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]) * 10 + np.random.rand(15))
    distance, k = get_distance(grey)
    best_n_cluster = get_elbow(distance)
    best_n_cluster




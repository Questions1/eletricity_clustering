import pandas as pd
from sklearn.cluster import KMeans
from numpy.random import random_sample
from math import log
from sklearn.datasets import load_iris


# returns series of random values sampled between min and max values of passed col
def get_rand_data(col):
    rng = col.max() - col.min()
    return pd.Series(random_sample(len(col))*rng + col.min())


def iter_kmeans(df, n_clusters, num_iters=5):
    rng = range(1, num_iters + 1)
    vals = pd.Series(index=rng)
    for i in rng:
        k = KMeans(n_clusters=n_clusters, n_init=3)
        k.fit(df)
        # print("Ref k: %s" % k.get_params()['n_clusters'])
        vals[i] = k.inertia_
    return vals


def gap_statistic(df, max_k=10):
    gaps = pd.Series(index=range(1, max_k + 1))
    for k in range(1, max_k + 1):
        km_act = KMeans(n_clusters=k, n_init=3)
        km_act.fit(df)

        # get ref dataset
        ref = df.apply(get_rand_data)
        ref_inertia = iter_kmeans(ref, n_clusters=k).mean()

        gap = log(ref_inertia - km_act.inertia_)

        # print("Ref: %s   Act: %s  Gap: %s" % (ref_inertia, km_act.inertia_, gap))
        gaps[k] = gap
    return gaps


def get_1_2(gaps):
    max = sorted(gaps)[-1]
    max_2 = sorted(gaps)[-2]
    max_index = gaps.index[gaps == max]
    max_2_index = gaps.index[gaps == max_2]
    return max_index, max_2_index


if __name__ == '__main__':
    iris = load_iris()
    iris_data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    iris_target = iris['target']

    gaps = gap_statistic(a)
    max_index, max_2_index = get_1_2(gaps)
    print(max_index[0], max_2_index[0])

km = KMeans(n_clusters=2)
km.fit(a)
km.
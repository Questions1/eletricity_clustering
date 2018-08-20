import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.cluster import KMeans


def load_data(df_raw):
    data = df_raw[['TimeStr', 'Ia']].copy()
    data.drop_duplicates('TimeStr', inplace=True)
    data['TimeStr'] = pd.to_datetime(data['TimeStr'])
    data.sort_values(by='TimeStr', inplace=True)

    return data


def fillin_timeseries(data):
    time_series = pd.date_range(data["TimeStr"].min(), data["TimeStr"].max(), freq='s')
    ts = pd.DataFrame(time_series, columns=["time"])

    data_new = data.merge(ts, how='right', left_on="TimeStr", right_on="time")
    data_new.drop(['TimeStr'], axis=1, inplace=True)
    data_new.sort_values(by='time', inplace=True)

    total_days = (data_new['time'].max() - data_new['time'].min()).total_seconds() // (24 * 3600)
    idx = data_new.index[data_new.index % (2 * total_days - 1) == 0]
    data_new = data_new.loc[idx, :]
    data_new.reset_index(drop=True, inplace=True)

    return data_new


def fillin_missvalue(data):
    data['Ia'] = data['Ia'].rolling(window=5, min_periods=1, center=False).median()

    return data


def find_cut(first_diff_abs, gap):
    idx, _ = signal.find_peaks(first_diff_abs, height=np.max(first_diff_abs) * 0.05)
    idx_new = np.append(np.append(0, idx), len(first_diff_abs))

    d1 = idx_new[:-2]
    d2 = idx_new[1:-1]
    d3 = idx_new[2:]

    d2_1 = d2 - d1
    d2_3 = d3 - d2

    the_max = d2_1
    the_max[d2_1 < d2_3] = d2_3[d2_1 < d2_3]
    cut_points = np.append(np.append(0, idx[the_max > gap]), len(first_diff_abs))

    return cut_points, idx


def cut_plot(data, cut_points):   # 之后再看看是否需要保存
    plt.plot(data['Ia'], linewidth=0.5, color='red')
    plt.vlines(cut_points, 0, np.max(data['Ia']), linewidth=0.5, color='g')
    plt.plot(first_diff_abs, linewidth=0.5, color='grey')


def get_label(data, cut_points):
    tmp = []  # 生成一段标签，用来表示所属的段
    for i in range(len(cut_points) - 1):
        tmp[cut_points[i]:cut_points[i + 1]] = [i] * (cut_points[i + 1] - cut_points[i])
    data['cut_points'] = tmp

    return data


def get_slice_feature(data):
    tmp = np.zeros(data.shape[0])
    tmp[idx] = 1
    data['idx'] = tmp

    indexs = np.unique(data['cut_points'])

    length = []
    for index in indexs:
        length.append(np.sum(data['cut_points'] == index))

    mean_ia = []
    for index in indexs:
        tmp = np.mean(data['Ia'][data['cut_points'] == index])
        mean_ia.append(tmp)

    peak_count = []
    for index in indexs:
        tmp = np.sum(data['idx'][data['cut_points'] == index])
        peak_count.append(tmp)

    accumulate_step = []
    for index in indexs:
        tmp = data['Ia'][data['cut_points'] == index].values
        d1 = tmp[:-1]
        d2 = tmp[1:]
        step = np.sum(np.abs(d2 - d1))
        accumulate_step.append(step)

    var = []
    for index in indexs:
        tmp = np.var(data['Ia'][data['cut_points'] == index])
        var.append(tmp)

    slice_feature = pd.DataFrame({'indexs': indexs,
                                  'length': length,
                                  'mean_ia': mean_ia,
                                  'peak_count': peak_count,
                                  'accumulate_step': accumulate_step,
                                  'var': var}, index=indexs)
    slice_feature['peak_count_ave'] = slice_feature['peak_count'] / slice_feature['length']
    slice_feature['accumulate_step_ave'] = slice_feature['accumulate_step'] / slice_feature['length']

    return slice_feature


def cluster_work_others(data, slice_feature, features, n):  # n indicates the number of classes
    kmean = KMeans(n_clusters=n)
    kmean.fit(slice_feature[features])
    slice_feature['first_labels'] = kmean.labels_

    first_labels = np.zeros(data.shape[0])

    labels = np.unique(kmean.labels_)
    for label in labels:
        label_x = slice_feature.index[slice_feature['first_labels'] == label]
        for i in label_x:
            first_labels[data['cut_points'] == i] = label
    data['first_labels'] = first_labels

    for label in labels:
        plt.scatter(data.loc[data['first_labels'] == label, 'TimeStr'].values,
                    data.loc[data['first_labels'] == label, 'Ia'].values, s=2)
        plt.xlim(data.TimeStr.min(), data.TimeStr.max())

    return data


if __name__ == '__main__':
    df_raw = pd.read_csv(r'./10.9.129.79.csv')
    data_1 = load_data(df_raw)
    data_2 = fillin_missvalue(data_1)
    data = fillin_missvalue(data_2)

    data.reset_index(inplace=True, drop=True)
    first_diff_abs = abs(data['Ia'].diff())

    cut_points, idx = find_cut(first_diff_abs, 300)
    data = get_label(data, cut_points)
    cut_plot(data, cut_points)

    slice_feature = get_slice_feature(data)

    data = cluster_work_others(data, slice_feature,
                               ['var', 'peak_count_ave', 'accumulate_step_ave'],
                               n=2)  # ['var', 'peak_count_ave', 'accumulate_step_ave']
    # 这块切的时候可以加窗来解决连续递增的情况








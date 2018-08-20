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


def fillin_missvalue(data, w):
    data['Ia'] = data['Ia'].rolling(window=w, min_periods=1, center=False).median()

    return data


def find_cut(first_diff_abs, gap, ratio):
    idx, _ = signal.find_peaks(first_diff_abs, height=np.max(first_diff_abs) * ratio)
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

    iqr = []
    for index in indexs:
        p_90 = np.percentile(data['Ia'][data['cut_points'] == index], 80)
        p_10 = np.percentile(data['Ia'][data['cut_points'] == index], 20)
        iqr.append(p_90 - p_10)

    slice_feature = pd.DataFrame({'indexs': indexs,
                                  'length': length,
                                  'mean_ia': mean_ia,
                                  'peak_count': peak_count,
                                  'accumulate_step': accumulate_step,
                                  'var': var,
                                  'iqr':iqr}, index=indexs)
    slice_feature['peak_count_ave'] = slice_feature['peak_count'] / slice_feature['length']
    slice_feature['accumulate_step_ave'] = slice_feature['accumulate_step'] / slice_feature['length']

    return slice_feature


def labeling_slice_feature(slice_feature, features):
    labeled_slice_feature = slice_feature.copy()
    for feature in features:
        label_name = '%s_label' % feature

        kmean = KMeans(n_clusters=2)
        kmean.fit(labeled_slice_feature[[feature]])

        center_0 = np.mean(labeled_slice_feature[feature][kmean.labels_ == 0])
        center_1 = np.mean(labeled_slice_feature[feature][kmean.labels_ == 1])
        if center_0 > center_1:
            tmp = np.abs(kmean.labels_ - 1)
            labeled_slice_feature[label_name] = tmp
        else:
            labeled_slice_feature[label_name] = kmean.labels_

    label_names = ['%s_label' % x for x in features]
    slice_sum_label = np.sum(labeled_slice_feature[label_names], axis=1)
    slice_sum_label[slice_sum_label > 0] = 1
    labeled_slice_feature['slice_sum_label'] = slice_sum_label

    return labeled_slice_feature


def cluster_work_others(data, labeled_slice_feature):
    the_labels = np.zeros(data.shape[0])

    label_x = labeled_slice_feature.index[labeled_slice_feature['slice_sum_label'] == 1]
    for i in label_x:
        the_labels[data['cut_points'] == i] = 1
    data['sum_label'] = the_labels

    return data


def plot_cluster(data):
    labels = np.unique(data['sum_label'])
    for label in labels:
        plt.scatter(data.index[data['sum_label'] == label].values,
                    data.loc[data['sum_label'] == label, 'Ia'].values, s=2)
        plt.vlines(cut_points, 0, np.max(data['Ia']), linewidth=0.5, color='g')


if __name__ == '__main__':
    df_raw = pd.read_csv(r'./10.15.203.11.csv')
    data_1 = load_data(df_raw)
    data_2 = fillin_missvalue(data_1, 10)
    data = fillin_missvalue(data_2, 10)

    data.reset_index(inplace=True, drop=True)
    first_diff_abs = abs(data['Ia'].diff())

    cut_points, idx = find_cut(first_diff_abs, 300, 0.05)
    data = get_label(data, cut_points)

    # cut_plot(data, cut_points)
    features = ['var', 'peak_count_ave', 'accumulate_step_ave', 'iqr']

    slice_feature = get_slice_feature(data)
    labeled_slice_feature = labeling_slice_feature(slice_feature, features)
    data = cluster_work_others(data, labeled_slice_feature)
    plot_cluster(data)









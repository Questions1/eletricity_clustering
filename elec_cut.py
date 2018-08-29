
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.cluster import KMeans
from sklearn import mixture


def load_data(df_raw):
    """
    数据去重
    把时间变换为python中的时间格式数据
    按照时间排序
    """
    data = df_raw[['TimeStr', 'Ia']].copy()
    data.drop_duplicates('TimeStr', inplace=True)
    data['TimeStr'] = pd.to_datetime(data['TimeStr'])
    data.sort_values(by='TimeStr', inplace=True)

    return data


def fillin_timeseries(data):
    """
    时间补全，确保每一天都是86400秒
    """
    time_series = pd.date_range(data["TimeStr"].min(), data["TimeStr"].max(), freq='s')
    ts = pd.DataFrame(time_series, columns=["time"])

    data_new = data.merge(ts, how='right', left_on="TimeStr", right_on="time")
    data_new.drop(['TimeStr'], axis=1, inplace=True)
    data_new.sort_values(by='time', inplace=True)
    data_new.reset_index(drop=True, inplace=True)

    return data_new


def fillin_missvalue(data, w):
    """
    中值滤波处理数据
    """
    data['Ia'] = data['Ia'].rolling(window=w, min_periods=1, center=False).median()

    return data


def find_cut(first_diff_abs, gap, ratio):
    """
    根据一阶差分找到切分点来切段，以备聚类
    """
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


def get_label(data, cut_points):
    """
    给切好段的时间序列段按照顺序打上类别标记0, 1, 2, ...
    """
    data_new = data.copy()
    tmp = []  # 生成一段标签，用来表示所属的段
    for i in range(len(cut_points) - 1):
        tmp[cut_points[i]:cut_points[i + 1]] = [i] * (cut_points[i + 1] - cut_points[i])
    data_new['cut_points'] = tmp

    return data_new


def get_slice_feature(data, idx):
    """
    对切好段的时间序列提取特征，以备聚类
    """
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

    slice_feature = pd.DataFrame({'length': length,
                                  'mean_ia': mean_ia,
                                  'peak_count': peak_count,
                                  'accumulate_step': accumulate_step,
                                  'var': var,
                                  'iqr':iqr}, index=indexs)
    slice_feature['peak_count_ave'] = slice_feature['peak_count'] / slice_feature['length']
    slice_feature['accumulate_step_ave'] = slice_feature['accumulate_step'] / slice_feature['length']

    return slice_feature


def labeling_slice_feature(slice_feature, features, times, vote):
    """
    根据slice_feature来对切好的段聚类，这个函数会用两轮，当然，略有区别(交集 | 并集)
    """
    if slice_feature.shape[0] < 2:
        print('_'*50 + 'stop')
        return
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
    if times == 1:
        slice_sum_label[slice_sum_label > 0] = 1
    elif times == 2:
        slice_sum_label[slice_sum_label < vote * len(features)] = 0
        slice_sum_label[slice_sum_label >= vote * len(features)] = 1
    else:
        print('error')
    labeled_slice_feature['slice_sum_label'] = slice_sum_label

    return labeled_slice_feature


def post_cluster(data, labeled_slice_feature):
    """
    把第一轮根据slice_feature聚类好的结果扩展到data上, 以备画图使用
    """
    labels = np.unique(labeled_slice_feature['slice_sum_label'])
    the_labels = np.zeros(data.shape[0])

    for label in labels:
        label_x = labeled_slice_feature.index[labeled_slice_feature['slice_sum_label'] == label]
        for i in label_x:
            the_labels[data['cut_points'] == i] = label
    data['sum_label'] = the_labels

    return data


def post_cluster_2(data, labeled_slice_feature, times, vote):
    """
    进行第二轮聚类
    """
    rest_slice_feature = labeled_slice_feature.loc[labeled_slice_feature['slice_sum_label'] == 0, features]
    rest_labeled_slice_feature = labeling_slice_feature(rest_slice_feature, features, times, vote)

    label_1 = labeled_slice_feature[['slice_sum_label']]
    label_2 = rest_labeled_slice_feature[['slice_sum_label']]
    label_12 = pd.merge(label_1, label_2, how='outer', left_index=True, right_index=True)
    label_12 = label_12.fillna(1)
    slice_sum_label = np.sum(label_12, axis=1)
    labeled_slice_feature['slice_sum_label'] = slice_sum_label

    return post_cluster(data, labeled_slice_feature)


def box(data_2, labeled_slice_feature):
    """
    对第二轮聚类之后得到的结果利用箱线图进行异常值处理，方面后续的GMM聚类
    """
    data_3 = data_2.copy()
    grey_indexs = labeled_slice_feature.index[labeled_slice_feature['slice_sum_label'] == 0]
    data_3_Ia = data_3['Ia'].copy()
    for grey_index in grey_indexs:
        tmp = data_3_Ia[data_3['cut_points'] == grey_index]
        Q1 = np.percentile(tmp, 25)
        Q3 = np.percentile(tmp, 75)
        IQR = Q3 - Q1
        up_bound = Q3 + IQR * 1.5
        low_bound = Q1 - IQR * 1.5
        tmp[tmp > up_bound] = up_bound
        tmp[tmp < low_bound] = low_bound
        data_3_Ia[data_2['cut_points'] == grey_index] = tmp
    data_3['Ia'] = data_3_Ia
    return data_3


def split_grey(data_3):
    """
    把灰色的点按照work_mean在分一下，只对下面的进行GMM聚类
    """
    work_mean = np.mean(data_3.loc[(data_3['sum_label'] > 0), 'Ia'])

    sum_label = data_3['sum_label'].copy()
    sum_label[(data_3['Ia'] > work_mean) & (data_3['sum_label'] == 0)] = 0.5
    data_4 = data_3.copy()
    data_4['sum_label'] = sum_label

    return data_4, work_mean


def plot_cluster(data):
    """
    对两轮聚类和一轮阈值判断的结果画图

    两类时:工作1(1, blue), 其他(0, grey)
    三类时:工作1(2, blue), 工作2(1, red), 其他(0, grey)
    四类时:工作1(2, blue), 工作2(1, red), 工作3(1.5, green) ,其他(0, grey)
    """
    labels = np.sort(np.unique(data['sum_label']))
    if len(labels) == 2:
        colors = ['grey', 'blue']
    elif len(labels) == 3:
        colors = ['grey', 'red', 'blue']
    elif len(labels) == 4:
        colors = ['grey', 'green', 'red', 'blue']
    else:
        print('error')
    for i in range(len(labels)):
        label = labels[i]
        plt.scatter(data.index[data['sum_label'] == label].values,
                    data.loc[data['sum_label'] == label, 'Ia'].values,
                    s=2, color=colors[i])
        plt.vlines(cut_points, 0, np.max(data['Ia']), linewidth=0.5, color='g')
    plt.show()


def get_distance(data):
    """
    引自elbow.py, 把数据聚成多个类别，输出各个类别下的组内方差和，画出"肘型图"
    """
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
    """
    引自elbow.py, 输出最佳聚类个数
    """
    a_1 = pd.Series(distance[1:])
    a_2 = pd.Series(distance[:-1])

    tmp = a_2 / a_1

    tmp_1 = pd.Series(tmp[1:].values)
    tmp_2 = pd.Series(tmp[:-1].values)

    tmp2 = tmp_2 / tmp_1
    return tmp2.values.argmax() + 2


def use_gaussian(data_4, grey, best_n_cluster, work_mean):
    """
    使用GMM对灰色的点进行聚类
    """
    grey['Ia'][grey['Ia'] > work_mean] = np.mean(grey['Ia'])
    grey['Ia'] = grey['Ia'] + np.random.normal(0, 0.3, len(grey['Ia']))
    clf = mixture.GaussianMixture(n_components=best_n_cluster, covariance_type='full')
    clf.fit(np.array(grey).reshape(-1, 1))
    predicted = clf.predict(np.array(grey).reshape(-1, 1))
    data_5 = data_4.copy()
    the_label = data_5['sum_label'].copy()
    the_label[the_label > 0] = best_n_cluster
    the_label[(the_label == 0) & (data_5['Ia'] > 0)] = predicted
    the_label[(the_label == 0) & (data_5['Ia'] == 0)] = -1
    data_5['sum_label'] = the_label

    return data_5, clf.means_, clf.covariances_, clf.weights_, clf


def plot_gaussian(data):
    """
    使用GMM聚类结束之后，画出聚类的结果
    """
    labels = np.sort(np.unique(data['sum_label']))
    plt.scatter(data.index[data['sum_label'] == best_n_cluster].values,
                data.loc[data['sum_label'] == best_n_cluster, 'Ia'].values,
                s=2, color='grey')
    for i in range(len(labels)-1):
        label = labels[i]
        plt.scatter(data.index[data['sum_label'] == label].values,
                    data.loc[data['sum_label'] == label, 'Ia'].values,
                    s=2)
    plt.vlines(cut_points, 0, np.max(data['Ia']), linewidth=0.5, color='g')


def get_idle(data, means_, m):
    """
    根据参数m动态地调整对待机的识别
    """
    tmp = pd.Series(means_.reshape(1, -1)[0]).rank()
    idle_labels = tmp.index.values[tmp <= m]
    data_6 = data.copy()
    sum_label = data_6['sum_label'].copy()

    sum_label[list(map(lambda x: x in idle_labels, data_6['sum_label']))] = 1
    sum_label[list(map(lambda x: x not in idle_labels, data_6['sum_label']))] = 2
    sum_label[data_6['Ia'] == 0] = 0
    data_6['sum_label'] = sum_label

    return data_6


def plot_final(data):
    """
    画出根据m动态调整后的图
    """
    colors = ['grey', 'green', 'red', 'blue', 'pink', 'yellow', 'cyan', 'orange', 'black']
    labels = np.sort(np.unique(data['sum_label']))
    for i in range(len(labels)):
        label = labels[i]
        plt.scatter(data.index[data['sum_label'] == label].values,
                    data.loc[data['sum_label'] == label, 'Ia'].values,
                    s=2, color=colors[np.int(label)])
    plt.vlines(cut_points, 0, np.max(data['Ia']), linewidth=0.5, color='g')


def get_out_para(means_, covariances_, weights_):
    """
    把模型参数输出，有两部分：

    一是GMM的模型参数，包括均值、方差、概率
    二是一个较为宽松的阈值
    """
    out_para = pd.DataFrame({'means': means_.reshape(1, -1)[0],
                             'cov': covariances_.reshape(1, -1)[0],
                             'alpha': weights_})
    up_index = out_para['means'].values.argmax()
    up_thre = out_para.loc[up_index, 'means'] + 3 * out_para.loc[up_index, 'cov']

    return out_para, up_thre


def predict(data_raw_1, best_n_cluster, up_thre, clf):
    """
    利用得到的聚类器(up_thre, clf)对原始数据(未经中值滤波处理)进行分类
    """
    label = np.zeros(data_raw_1.shape[0])
    label[data_raw_1['Ia'] > up_thre] = best_n_cluster
    left = np.array(data_raw_1['Ia'][data_raw_1['Ia'] <= up_thre]).reshape(-1, 1)
    label[data_raw_1['Ia'] <= up_thre] = clf.predict(left)
    data_raw_1['sum_label'] = label

    return data_raw_1


##
if __name__ == '__main__':
    ip_list = ['10.9.129.31', '10.9.129.30',
               # '10.9.129.175',  # 10.9.129.175的数据全是0
               '10.9.129.170',
               '10.9.129.171', '10.9.129.167',
               '10.9.129.79', '10.9.130.75', '10.9.129.96']

    # 数据预处理
    df_raw = pd.read_csv(r'./%s.csv' % ip_list[1])
    data_raw_1 = load_data(df_raw)
    data_raw_2 = fillin_missvalue(data_raw_1, 10)
    data_raw_3 = fillin_missvalue(data_raw_2, 10)
    data_raw_3.reset_index(inplace=True, drop=True)
    first_diff_abs = abs(data_raw_3['Ia'].diff())

    # 首先进行切块
    cut_points, idx = find_cut(first_diff_abs, 300, 0.05)
    data_0 = get_label(data_raw_3, cut_points)

    # 获取特征
    features = ['var', 'peak_count_ave', 'accumulate_step_ave', 'iqr']
    slice_feature = get_slice_feature(data_0, idx)

    # 第一轮聚类
    labeled_slice_feature = labeling_slice_feature(slice_feature, features, 1, 1)
    data_1 = post_cluster(data_0, labeled_slice_feature)
    plot_cluster(data_1)
    plt.close()

    # 第二轮聚类
    data_2 = post_cluster_2(data_1, labeled_slice_feature, 2, 0.5)
    data_3 = box(data_2, labeled_slice_feature)
    plot_cluster(data_3)
    plt.close()

    # 把灰色部分按照值来拆分成两部分，进一步判断待机所在区域
    data_4, work_mean = split_grey(data_3)
    plot_cluster(data_4)
    plt.close()

    # 根据"肘部图"获取最佳聚类个数, 不包括关机
    # 因为数据本身是离散的，得手动加一些扰动项
    grey = data_4.loc[(data_4['sum_label'] == 0) & (data_4['Ia'] > 0), ['Ia']]
    grey['Ia'][grey['Ia'] > work_mean] = np.mean(grey['Ia'])
    grey['Ia'] = grey['Ia'] + np.random.normal(0, 0.3, len(grey['Ia']))
    distance, k = get_distance(grey)
    plt.close()
    best_n_cluster = get_elbow(distance)

    # 使用高斯混合得到聚类中心, 并对数据做出预测, 画出图, 输出聚类器(up_thre, clf)
    data_5, means_, covariances_, weights_, clf = use_gaussian(data_4, grey, best_n_cluster, work_mean)
    out_para, up_thre = get_out_para(means_, covariances_, weights_)
    plot_gaussian(data_5)
    plt.hlines(out_para['means'], 0, data_5.shape[0])
    plt.close()

    # 下面动态地判断待机的GMM中心, 可以调整get_idle的最后一个参数，其最大取值为best_n_cluster
    data_6 = get_idle(data_5, means_, 2)
    plot_final(data_6)
    plt.close()

    # 这块存疑
    up_thre = work_mean

    # 基于聚类器(up_thre, clf)对原始数据(未经中值滤波处理)进行分类
    data_raw_1_label = predict(data_raw_1, best_n_cluster, up_thre, clf)
    plot_final(data_raw_1_label)
    plt.hlines(up_thre, 0, data_raw_1_label.shape[0])
    plt.close()

    # 下面动态地判断待机的GMM中心, 可以调整get_idle的最后一个参数m，其最大取值为best_n_cluster
    # 灰色关机，绿色待机，红色工作
    default_m = np.sum(out_para['means'] < 0.1) + 1
    data_final = get_idle(data_raw_1_label, means_, m=2)
    plot_final(data_final)
    plt.hlines(up_thre, 0, data_final.shape[0])


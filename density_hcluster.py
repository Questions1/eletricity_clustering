
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.cluster import KMeans
from sklearn import mixture
import statsmodels.nonparametric.api as smnp


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


##
if __name__ == '__main__':
    ip_list = ['10.9.129.31', '10.9.129.30',
               # '10.9.129.175',  # 10.9.129.175的数据全是0
               '10.9.129.170',
               '10.9.129.171', '10.9.129.167',
               '10.9.129.79', '10.9.130.75', '10.9.129.96']

    # 数据预处理
    df_raw = pd.read_csv(r'./%s.csv' % ip_list[-1])
    data_raw_1 = load_data(df_raw)
    data_raw_2 = fillin_missvalue(data_raw_1, 10)
    data_raw_3 = fillin_missvalue(data_raw_2, 10)
    data_raw_3.reset_index(inplace=True, drop=True)

    x_density = data_raw_3.Ia
    kde = smnp.KDEUnivariate(x_density)
    kde.fit(kernel='gau', bw='scott', fft=True, gridsize=100, cut=3.0, clip=(-np.inf, np.inf))
    x_plot, x_dens = kde.support, kde.density
    plt.plot(x_plot, x_dens)
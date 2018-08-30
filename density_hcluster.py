
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
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


def better_local(peaks, vallys, x_plot):
    if len(peaks) <= 1:
        print('One peak')
        return pd.DataFrame(columns=['x_plot', 'x_density']), np.nan
    new_vallys = []
    for i, _ in enumerate(peaks[:-1]):
        temp = vallys[(vallys > peaks[i]) & (vallys < peaks[i + 1])]
        new_vallys.append(int(np.min(temp)))

        x_plot = x_plot.reshape(-1)
        local_minmax = pd.DataFrame({'x': np.concatenate((x_plot[peaks], x_plot[new_vallys])),
                                     'y': np.concatenate((x_dens[peaks], x_dens[new_vallys]))})
        local_minmax.sort_values(by='x', inplace=True)
        local_minmax.reset_index(drop=True, inplace=True)

        return local_minmax


def get_sum_label(data_raw_2, local_min, m):
    final_data = data_raw_2.copy()
    threshold = local_min[m]
    sum_label = np.zeros(final_data.shape[0])

    sum_label[(final_data['Ia'] < 0.1)] = 0
    sum_label[(final_data['Ia'] >= 0.1) & (final_data['Ia'] < threshold)] = 1
    sum_label[final_data['Ia'] >= threshold] = 2
    final_data['sum_label'] = sum_label

    return final_data


def plot_final(data):
    """
    画出根据m动态调整后的图
    """
    colors = ['grey', 'green', 'red']
    for i in range(3):
        plt.scatter(data.index[data['sum_label'] == i].values,
                    data.loc[data['sum_label'] == i, 'Ia'].values,
                    s=2, color=colors[i])


##
if __name__ == '__main__':
    ip_list = ['10.9.129.31', '10.9.129.30',
               # '10.9.129.175',  # 10.9.129.175的数据全是0
               '10.9.129.170',
               '10.9.129.171', '10.9.129.167',
               '10.9.129.79', '10.9.130.75', '10.9.129.96']

    # 数据预处理
    df_raw = pd.read_csv(r'./%s.csv' % ip_list[0])
    data_raw_1 = load_data(df_raw)
    data_raw_2 = fillin_missvalue(data_raw_1, 10)
    data_raw_3 = fillin_missvalue(data_raw_2, 10)
    data_raw_3.reset_index(inplace=True, drop=True)

    x_density = data_raw_3.Ia
    kde = smnp.KDEUnivariate(x_density)
    kde.fit(kernel='gau', bw='scott', fft=True, gridsize=100, cut=3.0, clip=(-np.inf, np.inf))
    x_plot, x_dens = kde.support, kde.density

    peaks, p_heights = signal.find_peaks(x_dens, height=0.03 * max(x_dens))
    vallys, v_heights = signal.find_peaks(-x_dens)

    plt.plot(x_plot, x_dens)
    plt.scatter(x_plot[peaks], x_dens[peaks], color='red')
    plt.scatter(x_plot[vallys], x_dens[vallys], color='green')

    local_minmax = better_local(peaks, vallys, x_plot)

    plt.plot(x_plot, x_dens)
    plt.scatter(local_minmax.x, local_minmax.y)

    odd = list(map(lambda x: x % 2 == 1, local_minmax.index))
    local_min = local_minmax.x[odd].values

    final_data = get_sum_label(data_raw_2, local_min, m=1)
    plot_final(final_data)



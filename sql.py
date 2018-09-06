
import pyodbc
import pandas as pd
import os


def fetch_data(ip, date_begin, date_end):
    """
    取出某一个ip多日的数据
    """
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=10.19.7.60\WinCC;'
                          'DATABASE=ai_appliaction;UID=ai_user;PWD=1qaz#EDC')
    date_range = range(date_begin, date_end)
    data = pd.DataFrame()
    for date in date_range:
        sql = """SELECT IpAddr, Ia, TimeStr = CONVERT(VARCHAR(100), AddTime, 120) 
        FROM Tab_real_data_%s 
        WHERE IpAddr='%s'""" % (date, ip)
        try:
            tmp = pd.read_sql_query(sql, con=cnxn)
            data = data.append(tmp)
        except pd.io.sql.DatabaseError:
            print("no data in %s" % date)
    return data


if __name__ == '__main__':
    ip_list = ['10.9.129.31', '10.9.129.30',
               # '10.9.129.175',  # 10.9.129.175的数据全是0
               '10.9.129.170',
               '10.9.129.171', '10.9.129.167',
               '10.9.129.79', '10.9.130.75', '10.9.129.96']
    output_path = r'C:\Users\fan.dong\Desktop\project\elec_cut_v1.0'
    for ip in ip_list:
        print(ip)
        data_all = fetch_data(ip, 20180802, 20180831)
        data_all.to_csv(os.path.join(output_path, '%s.csv' % ip))

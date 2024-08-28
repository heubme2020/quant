import datetime
import os
import pandas as pd
import random
import numpy as np
import threading
import math
from tqdm import tqdm

not_standard_list = ['dividendRatio', 'pbInverse', 'debtToEquityRatio', 'quickRatio', 'ebitdaratio',
                     'operatingIncomeRatio','incomeBeforeTaxRatio', 'netIncomeRatio', 'grossProfitRatio',  'psInverse',
                     'cashFlowInverse', 'roe', 'dcfInverse', 'open', 'low', 'high', 'close', 'volume', 'marketCap',
                     'dayinyear', 'eps', 'epsdiluted', 'dividend', 'adjClose', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayoveryear']

exchange_list = ['Shenzhen', 'Shanghai', 'Swiss', 'Toronto', 'Johannesburg', 'Jakarta', 'Stockholm', 'Oslo', 'Tokyo',
                 'Saudi', 'Brussels', 'Arca', 'SaoPaulo', 'Thailand', 'Canadian', 'Australian', 'Korea', 'Warsaw',
                 'HongKong', 'London', 'NewYork', 'India', 'Taiwan', 'Moscow', 'Frankfurt', 'Milan', 'Nasdaq', 'Tlv', 'Otc']


#求取上个季度的最后一天
def get_pre_quarter_end_date(date_str):
    date_str = str(date_str)
    date_check = date_str[-4:]
    if date_check == '1231' or date_check == '0331' or date_check == '0630' or date_check =='0930':
        return date_str
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month-1)//3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str


def write_exchange_growth_death_data(groups, mean_data, std_data, daily_data):
    # 生成growth_death_train_data
    for i in tqdm(range(len(groups))):
        # print('growth_death_train_data_generating:' + str(i / (len(groups) + 1.0)))
        group = groups[i][1]
        # 去除最后3个季度的数据
        group = group.iloc[:-3]
        group = group.fillna(0).reset_index(drop=True)
        group_data_length = len(group)
        # 目前必须上市17个季度后才可以预测
        if group_data_length < 21:
            continue
        symbol = groups[i][0]
        group_daily = daily_data[(daily_data['symbol'] == symbol)].reset_index(drop=True)
        group_daily = group_daily.fillna(0)
        group_daily = group_daily[group_daily['adjClose'] > 0].reset_index(drop=True)
        if len(group_daily) < 1024:
            continue

        for j in range(21, group_data_length):
            endDate = group['endDate'].iloc[j - 4]
            data_basename = symbol + '_' + str(endDate) + '.h5'
            data_name = 'train/' + data_basename

            ##获取成长label和死亡label
            # 先获取截取日期
            endDate_datetime = datetime.datetime.strptime(str(endDate), "%Y%m%d")  # 将字符串转换为日期对象
            past_start = endDate_datetime - datetime.timedelta(days=512)
            past_start = int(past_start.strftime("%Y%m%d"))
            fore_end = endDate_datetime + datetime.timedelta(days=256)
            fore_end = int(fore_end.strftime("%Y%m%d"))
            # 再截取daily数据
            past_daily = group_daily[(group_daily['date'] > past_start) & (group_daily['date'] <= int(endDate))]
            past_daily = past_daily.reset_index(drop=True)
            if len(past_daily) < 256:
                continue
            fore_daily = group_daily[(group_daily['date'] > int(endDate)) & (group_daily['date'] <= fore_end)]
            fore_daily = fore_daily.reset_index(drop=True)
            if len(fore_daily) < 128:
                continue
            max_past = past_daily['adjClose'].max()
            median_past = past_daily['adjClose'].median()
            median_fore = fore_daily['adjClose'].median()
            min_fore = fore_daily['adjClose'].min()
            growth = float(median_fore) / float(max_past)
            if growth > 1024 or growth < -1024:
                continue
            if (growth >= -1.0/1024) and (growth <= 1.0/1024):
                continue
            death = float(median_past) / float(min_fore)
            if death > 1024 or death < -1024:
                continue
            if (death >= -1.0/1024) and (death <= 1.0/1024):
                continue
            # 下面进行归一化
            data = group.iloc[j - 21:j]
            data = data.reset_index()
            data.drop(columns=['index'], inplace=True)
            col_names = data.columns.values
            for k in range(2, len(data.columns)):
                col_name = col_names[k]
                if col_name in not_standard_list:
                    continue
                mean_value = mean_data.loc[mean_data['endDate'] == int(endDate), col_name].item()
                std_value = std_data.loc[std_data['endDate'] == int(endDate), col_name].item()
                data[col_name] = data[col_name] - mean_value
                if std_value != 0:
                    data[col_name] = data[col_name] / std_value
                else:
                    data[col_name] = 0

            data = data.fillna(0)
            data.replace([np.inf, -np.inf], 0, inplace=True)
            data = data.apply(lambda x: 0 if isinstance(x, (int, float)) and (
                        x > 1204 or x < -1024 or (-1.0 / 1024 < x < 1.0 / 1024)) else x)
            label = pd.DataFrame({
                'growth': [growth] * len(data),
                'death': [death] * len(data)
            })
            if np.isinf(label.values).any():
                continue
            data = pd.concat([data, label], axis=1)
            data.drop(columns=['symbol'], inplace=True)
            data = data.replace([np.inf, -np.inf], 0)
            data = data.astype('float32')
            data.to_hdf(data_name, key='data', mode='w')


def get_exchange_three_points_train_data(exchange):
    os.makedirs('train', exist_ok=True)
    # #获取growth_death_data_folder下已经存在的文件，避免重复生成
    # growth_death_files = [file for file in os.listdir('train') if file.endswith('.h5')]
    upper_exchange = exchange[0].upper() + exchange[1:]
    lower_exchange = exchange[0].lower() + exchange[1:]

    #读取数据
    daily_data = pd.read_csv('../data/' + upper_exchange + '/daily_' + exchange + '.csv')
    income_data = pd.read_csv('../data/' + upper_exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv('../data/' + upper_exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv('../data/' + upper_exchange + '/cashflow_' + exchange + '.csv')
    dividend_data = pd.read_csv('../data/' + upper_exchange + '/dividend_' + exchange + '.csv')
    mean_data = pd.read_csv('../data/' + upper_exchange + '/mean_' + exchange + '.csv')
    std_data = pd.read_csv('../data/' + upper_exchange + '/std_' + exchange + '.csv')
    #合并财务相关数据
    financial_data = pd.merge(dividend_data, income_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    financial_data = financial_data.dropna()
    financial_data = financial_data.reset_index(drop=True)
    financial_data['endDate'] = financial_data['endDate'].apply(get_pre_quarter_end_date)
    financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    financial_data = financial_data.reset_index(drop=True)
    #删除股票数为0的行
    financial_data = financial_data[financial_data['numberOfShares'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)
    #删除totalStockholdersEquity为0的行
    financial_data = financial_data[financial_data['totalStockholdersEquity'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)
    #删除totalCurrentLiabilities为0的行
    financial_data = financial_data[financial_data['totalCurrentLiabilities'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)

    #删除financial_data中endDate小于mean std data中最早的endDate的所有行
    financial_data = financial_data[financial_data['endDate'].astype(float) >= mean_data['endDate'].iloc[0]]
    financial_data = financial_data.reset_index(drop=True)

    # 删除financial_data中endDate小于mean std data中最早的endDate的所有行
    financial_data = financial_data[financial_data['endDate'] >= mean_data['endDate'].iloc[0].astype(str)]
    financial_data = financial_data.reset_index(drop=True)

    groups = list(financial_data.groupby('symbol'))
    random.shuffle(groups)
    group_count = len(groups)
    split_count = int(0.2*group_count)
    groups_0 = groups[:split_count]
    groups_1 = groups[split_count:2*split_count]
    groups_2 = groups[2*split_count:3*split_count]
    groups_3 = groups[3*split_count:4*split_count]
    groups_4 = groups[4*split_count:]

    # 创建线程并启动它们
    thread0 = threading.Thread(target=write_exchange_growth_death_data, args=(groups_0, mean_data, std_data, daily_data))
    thread1 = threading.Thread(target=write_exchange_growth_death_data, args=(groups_1, mean_data, std_data, daily_data))
    thread2 = threading.Thread(target=write_exchange_growth_death_data, args=(groups_2, mean_data, std_data, daily_data))
    thread3 = threading.Thread(target=write_exchange_growth_death_data, args=(groups_3, mean_data, std_data, daily_data))
    thread4 = threading.Thread(target=write_exchange_growth_death_data, args=(groups_4, mean_data, std_data, daily_data))

    thread0.start()
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    thread0.join()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()


def get_three_points_train_data(exchanges):
    random.shuffle(exchanges)
    os.makedirs('train', exist_ok=True)

    for exchange in exchanges:
        print(exchange)
        get_exchange_three_points_train_data(exchange)


def write_growth_death_data(file_list):
    csv_name = 'growth_death_train_data_17.csv'
    if os.path.exists(csv_name):
        os.remove(csv_name)
    basename = os.path.basename(file_list[0])
    basename_splits = basename.split('_')
    symbol = basename_splits[0]
    endDate_splits = basename_splits[1].split('.')
    endDate = float(endDate_splits[0])
    data = pd.read_hdf(file_list[0])
    growth = data['growth'].iloc[-4]
    death = data['death'].iloc[-4]
    input_data = data.iloc[:-4, 1:-2]
    input_data = input_data.T
    input_data = np.ravel(input_data.values)
    data = pd.DataFrame(input_data)
    data = data.T
    data['growth'] = growth
    data['death'] = death
    data.insert(0, 'endDate', endDate)
    data.insert(0, 'symbol', symbol)
    data.to_csv(csv_name, mode='a', header=True, index=False)

    for i in tqdm(range(1, len(file_list))):
        file = file_list[i]
        basename = os.path.basename(file)
        basename_splits = basename.split('_')
        symbol = basename_splits[0]
        endDate_splits = basename_splits[1].split('.')
        endDate = float(endDate_splits[0])
        data = pd.read_hdf(file)
        growth = data['growth'].iloc[-4]
        death = data['death'].iloc[-4]
        input_data = data.iloc[:-4, 1:-2]
        input_data = input_data.T
        input_data = np.ravel(input_data.values)
        data = pd.DataFrame(input_data)
        data = data.T
        data['growth'] = growth
        data['death'] = death
        data.insert(0, 'endDate', endDate)
        data.insert(0, 'symbol', symbol)
        data.to_csv(csv_name, mode='a', header=False, index=False)


def convert_data():
    train_h5_files = [os.path.join('train', f) for f in os.listdir('train') if f.endswith('.h5')]

    random.shuffle(train_h5_files)
    write_growth_death_data(train_h5_files)


if __name__ == '__main__':
    get_three_points_train_data(exchange_list)
    convert_data()

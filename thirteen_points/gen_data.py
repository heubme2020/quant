import datetime
import os
import pandas as pd
from tqdm import tqdm
import random
from dateutil.relativedelta import relativedelta
import numpy as np
import threading

not_standard_list = ['open', 'low', 'high', 'close', 'adjClose', 'volume','dividendRatio', 'pbInverse', 'psInverse',
                     'cashFlowInverse', 'debtInverse', 'totalStockholdersEquityInverse', 'quickRatioInverse',
                     'roeInverse', 'interestInverse', 'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayinyear', 'dayoveryear']

exchange_list = ['Shenzhen', 'Shanghai', 'Swiss', 'Toronto', 'Johannesburg', 'Jakarta', 'Stockholm', 'Oslo', 'Tokyo',
                 'Saudi', 'Brussels', 'Arca', 'SaoPaulo', 'Thailand', 'Canadian', 'Australian', 'Korea', 'Warsaw',
                 'HongKong', 'London', 'NewYork', 'India', 'Taiwan', 'Moscow', 'Frankfurt', 'Milan', 'Nasdaq', 'Tlv', 'Otc']

candidate_exchanges = ['SIX', 'SHZ', 'TSX', 'JNB', 'NASDAQ', 'JKT', 'STO', 'PNK', 'OSL', 'JPX', 'TLV', 'SAU',
                       'EURONEXT', 'AMEX', 'OTC', 'SHH', 'SAO', 'SET', 'CNQ', 'ASX', 'KSC', 'WSE', 'HKSE', 'LSE',
                       'NYSE', 'NSE', 'TAI', 'MCX', 'XETRA', 'MIL']

exchange_reference_dict = {'Shenzhen':['SZ', 'SHZ', 'SZSE', 'Shenzhen'],
                           'Shanghai':['SS', 'SHH', 'SSE', 'Shanghai'],
                           'Swiss':['SW', 'SIX', 'SWX', 'Swiss Exchange'],
                           'Toronto':['TO', 'TSX', 'TSX', 'Toronto'],
                           'Johannesburg':['JO', 'JNB', 'JSE', 'Johannesburg'],
                           'Jakarta':['JK', 'JKT', 'JKSE', 'Jakarta Stock Exchange'],
                           'Stockholm':['ST', 'STO', 'STO', 'Stockholm Stock Exchange'],
                           'Oslo':['OL', 'OSL', 'OSE', 'Oslo Stock Exchange'],
                           'Tokyo':['T', 'JPX', 'JPX', 'Tokyo'],
                           'Saudi':['SR', 'SAU', 'Tadawul', 'Saudi'],
                           'Brussels':['BR', 'EURONEXT', 'Euronext', 'Brussels'],
                           'Arca':['', 'AMEX', 'Arca', 'New York Stock Exchange Arca'],
                           'SaoPaulo':['SA', 'SAO', 'Bovespa', 'São Paulo'],
                           'Thailand':['BK', 'SET', 'SET', 'Thailand'],
                           'Canadian':['CN', 'CNQ', 'CSE', 'Canadian Sec'],
                           'Australian':['AX', 'ASX', 'ASX', 'Australian Securities Exchange'],
                           'Korea':['KS', 'KSC', 'KSE', 'KSE'],
                           'Warsaw':['WA', 'WSE', 'WSE', 'Warsaw Stock Exchange'],
                           'HongKong':['HK', 'HKSE', 'HKEx', 'HKSE'],
                           'London':['L', 'LSE', 'LSE', 'London Stock Exchange'],
                           'NewYork':['', 'NYSE', 'NYSE', 'New York Stock Exchange'],
                           'India':['NS', 'NSE', 'NSE', 'National Stock Exchange of India'],
                           'Taiwan':['TW', 'Tai', 'TWSE', 'Taiwan'],
                           'Moscow':['ME', 'MCX', 'MOEX', 'MCX'],
                           'Frankfurt':['DE', 'XETRA', 'FWB', 'Frankfurt Stock Exchange'],
                           'Milan':['MI', 'MIL', 'MIL', 'Milan'],
                           'Nasdaq':['', 'NASDAQ', 'NASDAQ', 'Nasdaq'],
                           'Tlv':['TA', 'TLV', 'TASE', 'Tel Aviv'],
                           'Otc':['', 'OTC', 'OTC', 'Other OTC']}


def date_back(date_str):
    date_str = str(date_str)
    str_list = list(date_str)  # 字符串转list
    str_list.insert(4, '-')  # 在指定位置插入字符串
    str_list.insert(7, '-')  # 在指定位置插入字符串
    str_out = ''.join(str_list)  # 空字符连接
    return str_out

def day_in_year(date):
    date = str(date)
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    date = datetime.date(year, month, day)
    idx = int(date.strftime('%j'))
    return idx


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


#计算上市以来的天数
def day_over_year(date, first_date):
    date = str(date)
    first_date = str(first_date)
    date_format = "%Y%m%d"
    # 将日期字符串转换为日期类型
    date = datetime.datetime.strptime(date, date_format)
    first_date = datetime.datetime.strptime(first_date, date_format)
    # 计算相差的天数
    diff = date - first_date
    # 返回相差的天数，使用abs函数确保结果是正数
    return abs(diff.days)


def get_data_input(data):
    data_past = data.iloc[:1201].reset_index(drop=True)
    data_fore = data.iloc[1201:].reset_index(drop=True)
    growth = data['growth'].iloc[0]
    death = data['death'].iloc[0]
    financial_past = data_past.iloc[:, 1:102].drop_duplicates().reset_index(drop=True)
    financial_past = financial_past.iloc[-17:].reset_index(drop=True)
    financial_fore = data_fore.iloc[:, 1:102].drop_duplicates().reset_index(drop=True)
    financial_fore = financial_fore.drop(financial_fore.index[0]).reset_index(drop=True)
    financial_fore = financial_fore.iloc[:7].reset_index(drop=True)
    data_past = data_past.iloc[:, :124].reset_index(drop=True)

    return data_past, financial_past, financial_fore, growth, death


def gen_group_train_data(groups, daily_data, mean_data, std_data):
    days_input = 1201
    days_output = 599
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        # 去除最后7个季度的数据
        group = group.iloc[:-7]
        if len(group) < 29:
            continue
        symbol = groups[i][0]
        group = group.reset_index()
        group.drop(columns=['index'], inplace=True)
        group['endDate'] = group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(group)
        for j in range(sheet_count - 1):
            dates = pd.date_range(start=group['endDate'].iloc[j], end=group['endDate'].iloc[j + 1])
            dates = dates.date.tolist()
            for k in range(1, len(dates) - 1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = group.iloc[j].copy()
                data_append['endDate'] = date_str
                group = pd.concat([group, data_append.to_frame().T], ignore_index=True)
        date = group['endDate'].iloc[sheet_count - 1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_quarter_end = date + relativedelta(months=3)
        next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
        dates = pd.date_range(start=group['endDate'].iloc[sheet_count - 1], end=next_quarter_end_str)
        dates = dates.date.tolist()
        for m in range(1, len(dates) - 1):
            date = dates[m]
            date_str = date.strftime('%Y-%m-%d')
            data_append = group.iloc[sheet_count - 1].copy()
            data_append['endDate'] = date_str
            group = pd.concat([group, data_append.to_frame().T], ignore_index=True)
        group['endDate'] = group['endDate'].apply(lambda x: x.replace('-', ''))
        group.sort_values(by='endDate', inplace=True, ascending=True)
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        group = group.rename(columns={'endDate': 'date'})
        group['date'] = group['date'].apply(lambda x: x.replace('-', ''))
        group['date'] = group['date'].astype(int)
        # 合并daily信息
        daily_group = daily_data[daily_data['symbol'] == symbol]
        daily_group = daily_group.reset_index(drop=True)
        if len(daily_group) < 1800:
            continue
        # 先获得第一天的close和adjClose价格和date
        first_close = daily_group['close'].iloc[0]
        if first_close == 0:
            continue
        first_adjClose = daily_group['adjClose'].iloc[0]
        if first_adjClose == 0:
            continue
        first_date = daily_group['date'].iloc[0]

        group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol']).reset_index(drop=True)
        group = group.fillna(0).reset_index(drop=True)
        group = group[group['close'] > 0].reset_index(drop=True)
        group = group[group['adjClose'] > 0].reset_index(drop=True)
        group = group[group['numberOfShares'] > 0].reset_index(drop=True)
        group = group[group['totalAssets'] > 0].reset_index(drop=True)
        if len(group) < 1800:
            continue

        # 先添加indicator相关一些
        # 添加分红率这一列
        group['dividendRatio'] = group['dividend'] / group['close']
        # 添加每股净资产换成市净率的倒数
        group['pbInverse'] = group['netAssetValuePerShare'] / group['close']
        # 添加每股销售额换成市销率的倒数
        group['psInverse'] = group['revenuePerShare'] / group['close']
        # 每股经营现金流换成对应的，我也不知道叫啥
        group['cashFlowInverse'] = group['cashFlowPerShare'] / group['close']
        # 每股的负债额换成对应的
        group['debtInverse'] = group['totalDebt'] / (group['numberOfShares'] * group['close'])
        # 每股的totalStockholdersEquity对应的，也叫市净率
        group['totalStockholdersEquityInverse'] = group['totalStockholdersEquity'] / (group['numberOfShares'] * group['close'])
        # quickRatioInverse
        group['quickRatioInverse'] = group['totalStockholdersEquityInverse'] * group['quickRatio']
        # roeInverse
        group['roeInverse'] = group['totalStockholdersEquityInverse'] * group['roe']
        # 还有其它几个
        # interestInverse
        group['interestInverse'] = group['interestExpense'] / (group['numberOfShares'] * group['close'])
        # cashInverse
        group['cashInverse'] = group['cashAndCashEquivalents'] / (group['numberOfShares'] * group['close'])
        # grossInverse
        group['grossInverse'] = group['grossProfit'] / (group['numberOfShares'] * group['close'])
        # 处理其它几个价格
        group['delta'] = group['high'] - group['low']
        group['open'] = group['open'] / group['close']
        group['low'] = group['low'] / group['close']
        group['high'] = group['high'] / group['close']
        group['close'] = group['close'] / first_close
        group['adjClose'] = group['adjClose'] / first_adjClose
        # 换手率%
        group['turnoverRate'] = group['volume'] / group['numberOfShares']
        # 交易市值与净资产的比值%

        group['turnoverNetAssetRatio'] = group['volume'] * group['close'] / (
                group['totalAssets'] - group['totalDebt'])
        # 把volume这一列做下处理
        group['volume'] = group['volume'].pct_change().fillna(0)
        # 加入date在一年中位置这一列
        group['dayinyear'] = group['date']
        group['dayinyear'] = group['dayinyear'].apply(lambda x: day_in_year(x))
        group['dayinyear'] = group['dayinyear'] / 365.0
        # 加入上市以来的日期
        group['dayoveryear'] = group['date']
        group['dayoveryear'] = group['dayoveryear'].apply(lambda x: day_over_year(x, first_date))
        group['dayoveryear'] = group['dayoveryear'] / 365.0

        # 然后再删除indicator相关的列
        group.drop(columns='netAssetValuePerShare', inplace=True)
        group.drop(columns='revenuePerShare', inplace=True)
        group.drop(columns='debtToEquityRatio', inplace=True)
        group.drop(columns='cashFlowPerShare', inplace=True)
        group.drop(columns='quickRatio', inplace=True)
        group.drop(columns='earningsMultiple', inplace=True)
        group.drop(columns='operatingMargin', inplace=True)
        group.drop(columns='pretaxProfitMargin', inplace=True)
        group.drop(columns='netProfitMargin', inplace=True)
        group.drop(columns='grossProfitMargin', inplace=True)
        group.drop(columns='roe', inplace=True)
        group.drop(columns='dcfPerShare', inplace=True)
        group = group.fillna(0).reset_index(drop=True)
        group = group[group['close'] > 0].reset_index(drop=True)
        group = group[group['adjClose'] > 0].reset_index(drop=True)
        if len(group) < 1800:
            continue
        # print(group)
        # 我们用1800天的数据进行训练，1201天作为输入，599作为输出
        group_data_length = len(group)
        group_23_24_list = []
        for j in range(days_input, group_data_length - days_output):
            date_ref = group['date'].iloc[j]
            data_basename = symbol + '_' + str(date_ref) + '.h5'
            data_name = 'train/' + data_basename
            date_object = datetime.datetime.strptime(str(date_ref), "%Y%m%d")
            weekday = date_object.weekday()
            date_day = date_object.day
            if date_day >= 23 and weekday == 4:  #如果是23号后的周五，就保存数据；3个点的策略一季度更新股票此，13个点一个月更新合理
                #如果23，24号的数据已经有了，就不存30，31号的了
                if date_day == 30:
                    date_format = "%Y%m%d"
                    date_23 = datetime.datetime.strptime(str(date_ref), date_format)
                    date_23 = date_23 - datetime.timedelta(days=7)
                    data_basename_23 = symbol + '_' + str(date_23.strftime("%Y%m%d")) + '.h5'
                    data_name_23 = 'train/' + data_basename_23
                    # print(data_basename_23)
                    if data_name_23 in group_23_24_list:
                        # print(data_basename_23 + ' existed!')
                        continue
                if date_day == 31:
                    date_format = "%Y%m%d"
                    date_24 = datetime.datetime.strptime(str(date_ref), date_format)
                    date_24 = date_24 - datetime.timedelta(days=7)
                    data_basename_24 = symbol + '_' + str(date_24.strftime("%Y%m%d")) + '.h5'
                    data_name_24 = 'train/' + data_basename_24
                    # print(data_basename_24)
                    if data_name_24 in group_23_24_list:
                        # print(data_basename_24 + ' existed!')
                        continue
                data = group.iloc[j - days_input:j + days_output].reset_index(drop=True)
                if len(data) != 1800:
                    continue
                col_names = data.columns.values
                # 获得上个的季度末日期
                date_str = datetime.datetime.strptime(str(date_ref), '%Y%m%d').date()
                quarter_month = ((date_str.month - 1) // 3) * 3 + 1
                quarter_end_date = datetime.date(date_str.year, quarter_month, 1) + datetime.timedelta(days=-1)
                endDate = int(quarter_end_date.strftime('%Y%m%d'))
                for k in range(2, len(col_names)):
                    col_name = col_names[k]
                    #如果不在not_standard_list列表，则进行归一化
                    if col_name not in not_standard_list:
                        mean_value = mean_data.loc[mean_data['endDate'] == endDate, col_name].item()
                        std_value = std_data.loc[std_data['endDate'] == endDate, col_name].item()
                        data[col_name] = data[col_name] - mean_value
                        if std_value != 0:
                            data[col_name] = data[col_name] / std_value
                        else:
                            data[col_name] = 0

                # 先获取截取日期
                past_start = date_object - datetime.timedelta(days=512)
                past_start = int(past_start.strftime("%Y%m%d"))
                fore_end = date_object + datetime.timedelta(days=512)
                fore_end = int(fore_end.strftime("%Y%m%d"))
                # 再截取daily数据
                past_daily = data[(data['date'] > past_start) & (data['date'] <= int(date_ref))]
                past_daily = past_daily.reset_index(drop=True)
                if len(past_daily) < 256:
                    continue
                fore_daily = data[(data['date'] > int(date_ref)) & (data['date'] <= fore_end)]
                fore_daily = fore_daily.reset_index(drop=True)
                if len(fore_daily) < 256:
                    continue
                max_past = past_daily['adjClose'].max()
                median_past = past_daily['adjClose'].median()
                median_fore = fore_daily['adjClose'].median()
                min_fore = fore_daily['adjClose'].min()
                growth = float(median_fore) / float(max_past)
                if growth > 1024 or growth < 1.0/1024:
                    continue
                death = float(median_past) / float(min_fore)
                if death > 1024 or death < 1.0/1024:
                    continue
                label = pd.DataFrame({
                    'growth': [growth] * len(data),
                    'death': [death] * len(data)
                })
                if np.isinf(label.values).any():
                    continue
                data = data.fillna(0)
                data = data.replace([np.inf, -np.inf], 0)
                data = data.apply(lambda x: 0 if isinstance(x, (int, float)) and (x > 1204 or x < -1024) else x)
                data = pd.concat([data, label], axis=1)
                data.drop(columns=['symbol'], inplace=True)
                data_past, financial_past, financial_fore, growth, death = get_data_input(data)
                if len(data_past) != 1201:
                    continue
                if len(financial_past) != 17:
                    continue
                if len(financial_fore) != 7:
                    continue
                data = data.astype('float32')
                data.to_hdf(data_name, key='data', mode='w')
                if date_day == 23 or date_day == 24:
                    group_23_24_list.append(data_name)


def gen_exchange_thirteen_points_train_data(exchange):
    print(exchange)
    os.makedirs('train', exist_ok=True)
    upper_exchange = exchange[0].upper() + exchange[1:]
    lower_exchange = exchange[0].lower() + exchange[1:]
    # 加载数据
    daily_data = pd.read_csv('../data/' + upper_exchange + '/daily_' + exchange + '.csv')
    income_data = pd.read_csv('../data/' + upper_exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv('../data/' + upper_exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv('../data/' + upper_exchange + '/cashflow_' + exchange + '.csv')
    indicator_data = pd.read_csv('../data/' + upper_exchange + '/indicator_' + exchange + '.csv')
    mean_data = pd.read_csv('../data/' + upper_exchange + '/mean_' + exchange + '.csv')
    std_data = pd.read_csv('../data/' + upper_exchange + '/std_' + exchange + '.csv')


    # 合并财务相关数据
    financial_data = pd.merge(indicator_data, income_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    financial_data = financial_data.dropna()
    financial_data = financial_data.reset_index(drop=True)
    financial_data['endDate'] = financial_data['endDate'].apply(get_pre_quarter_end_date)
    financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    financial_data = financial_data.reset_index(drop=True)
    # 删除股票数为0的行
    financial_data = financial_data[financial_data['numberOfShares'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)
    # 删除totalStockholdersEquity为0的行
    financial_data = financial_data[financial_data['totalStockholdersEquity'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)
    # 删除totalCurrentLiabilities为0的行
    financial_data = financial_data[financial_data['totalCurrentLiabilities'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)

    # 删除financial_data中endDate小于mean std data中最早的endDate的所有行
    financial_data = financial_data[financial_data['endDate'].astype(float) >= mean_data['endDate'].iloc[0]]
    financial_data = financial_data.reset_index(drop=True)

    groups = list(financial_data.groupby('symbol'))
    random.shuffle(groups)
    group_count = len(groups)
    split_count = int(0.2*group_count)
    groups_0 = groups[:split_count]
    groups_1 = groups[split_count:split_count*2]
    groups_2 = groups[split_count*2:split_count*3]
    groups_3 = groups[split_count*3:split_count*4]
    groups_4 = groups[split_count*4:]

    # 创建线程并启动它们
    thread0 = threading.Thread(target=gen_group_train_data, args=(groups_0, daily_data, mean_data, std_data))
    thread1 = threading.Thread(target=gen_group_train_data, args=(groups_1, daily_data, mean_data, std_data))
    thread2 = threading.Thread(target=gen_group_train_data, args=(groups_2, daily_data, mean_data, std_data))
    thread3 = threading.Thread(target=gen_group_train_data, args=(groups_3, daily_data, mean_data, std_data))
    thread4 = threading.Thread(target=gen_group_train_data, args=(groups_4, daily_data, mean_data, std_data))

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


def write_index(index, file, offset):
    with pd.HDFStore(file, 'r') as input_store:
        df = input_store.get('data')  # 假设每个HDF5文件中的DataFrame键为'df'
        data = df.values.astype(np.float32)
        num_rows, num_cols = data.shape
        entry = f'{offset}\t{num_rows}\t{num_cols}\n'
        offset += num_rows * num_cols * 4  # float32 has 4 bytes
    return entry, offset, data


def convert_data():
    h5_files = [os.path.join('train', f) for f in os.listdir('train') if f.endswith('.h5')]
    random.shuffle(h5_files)
    #0.7做训练集，0.3做验证集
    train_num = int(0.7*len(h5_files))
    train_files = h5_files[:train_num]
    validate_files = h5_files[train_num:]

    offset = 0
    output_file = 'train_data.bin'
    if os.path.exists(output_file):
        os.remove(output_file)
    train_index_file = 'train_index.txt'
    if os.path.exists(train_index_file):
        os.remove(train_index_file)
    val_index_file = 'validate_index.txt'
    if os.path.exists(val_index_file):
        os.remove(val_index_file)

    data_first = pd.read_hdf(train_files[0])
    pd.DataFrame(data_first.columns).to_csv('columns.csv', index=False, header=False)
    # 合并数据到二进制文件并生成索引
    with open(output_file, 'wb') as bin_file:
        with open(train_index_file, 'w') as train_idx:
            for file in tqdm(train_files):
                entry, offset, data = write_index(train_idx, file, offset)
                bin_file.write(data.tobytes())
                train_idx.write(entry)

        with open(val_index_file, 'w') as val_idx:
            for file in tqdm(validate_files):
                entry, offset, data = write_index(val_idx, file, offset)
                bin_file.write(data.tobytes())
                val_idx.write(entry)


def gen_thirteen_points_train_data(exchanges):
    random.shuffle(exchanges)
    os.makedirs('train', exist_ok=True)

    for exchange in exchanges:
        gen_exchange_thirteen_points_train_data(exchange)


if __name__ == '__main__':
    gen_thirteen_points_train_data(exchange_list)
    convert_data()

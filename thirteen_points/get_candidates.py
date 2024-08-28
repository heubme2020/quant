import os
import torch
import numpy as np
import random
import pandas as pd
import datetime
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import threading

not_standard_list = ['open', 'low', 'high', 'close', 'adjClose', 'volume','dividendRatio', 'pbInverse', 'psInverse',
                     'cashFlowInverse', 'debtInverse', 'totalStockholdersEquityInverse', 'quickRatioInverse',
                     'roeInverse', 'interestInverse', 'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayinyear', 'dayoveryear']


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


def get_groups_candidates(groups, model, daily_data, mean_data, std_data, thread_idx):
    device = 'cuda'
    score_data = pd.DataFrame()
    #生成financial_forecast_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        symbol = groups[i][0]
        group = group.reset_index()
        group.drop(columns=['index'], inplace=True)
        group['endDate'] = group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(group)
        for j in range(sheet_count-1):
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
        #合并daily信息
        daily_group = daily_data[daily_data['symbol'] == symbol]
        daily_group = daily_group.reset_index(drop=True)
        if len(daily_group) < 1201:
            continue
        #先获得第一天的close和adjClose价格和date
        first_close = daily_group['close'].iloc[0]
        first_adjClose = daily_group['adjClose'].iloc[0]
        first_date = daily_group['date'].iloc[0]

        group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol'])
        group = group.fillna(0).reset_index(drop=True)
        group = group[group['close'] > 0].reset_index(drop=True)
        group = group[group['adjClose'] > 0].reset_index(drop=True)
        group = group[group['numberOfShares'] > 0].reset_index(drop=True)
        if len(group) < 1201:
            continue

        #先添加indicator相关一些
        #添加分红率这一列
        group['dividendRatio'] = group['dividend']/ group['close']
        #添加每股净资产换成市净率的倒数
        group['pbInverse'] = group['netAssetValuePerShare'] / group['close']
        #添加每股销售额换成市销率的倒数
        group['psInverse'] = group['revenuePerShare'] / group['close']
        #每股经营现金流换成对应的，我也不知道叫啥
        group['cashFlowInverse'] = group['cashFlowPerShare']/ group['close']
        #每股的负债额换成对应的
        group['debtInverse'] = group['totalDebt']/(group['numberOfShares']*group['close'])
        #每股的totalStockholdersEquity对应的，也叫市净率
        group['totalStockholdersEquityInverse'] = group['totalStockholdersEquity']/\
                                                  (group['numberOfShares']*group['close'])
        #quickRatioInverse
        group['quickRatioInverse'] = group['totalStockholdersEquityInverse']*group['quickRatio']
        #roeInverse
        group['roeInverse'] = group['totalStockholdersEquityInverse']*group['roe']
        #还有其它几个
        #interestInverse
        group['interestInverse'] = group['interestExpense']/(group['numberOfShares']*group['close'])
        #cashInverse
        group['cashInverse'] = group['cashAndCashEquivalents']/(group['numberOfShares']*group['close'])
        #grossInverse
        group['grossInverse'] = group['grossProfit']/(group['numberOfShares']*group['close'])
        #处理其它几个价格
        group['delta'] = group['high'] - group['low']
        group['open'] = group['open'] / group['close']
        group['low'] = group['low'] / group['close']
        group['high'] = group['high'] / group['close']
        group['close'] = group['close']/first_close
        group['adjClose'] = group['adjClose']/first_adjClose
        #换手率%
        group['turnoverRate'] = group['volume']/group['numberOfShares']
        #交易市值与净资产的比值%
        group['turnoverNetAssetRatio'] = group['volume']*group['close']/(group['totalAssets'] - group['totalDebt'] + 0.00001)
        #把volume这一列做下处理
        group['volume'] = group['volume'].pct_change().fillna(0)
        #删除
        #加入date在一年中位置这一列
        group['dayinyear'] = group['date']
        group['dayinyear'] = group['dayinyear'].apply(lambda x: day_in_year(x))
        group['dayinyear'] = group['dayinyear']/365.0
        #加入上市以来的日期
        group['dayoveryear'] = group['date']
        group['dayoveryear'] = group['dayoveryear'].apply(lambda x: day_over_year(x, first_date))
        group['dayoveryear'] = group['dayoveryear']/365.0

        #然后再删除indicator相关的列
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
        group = group[group['adjClose'] > 0].reset_index(drop=True)
        data = group.iloc[-1201:].reset_index(drop=True)
        if len(data) != 1201:
            continue
        # print(financial_past)
        col_names = data.columns.values
        for k in range(2, len(col_names)):
            col_name = col_names[k]
            #如果不在not_standard_list列表，则进行归一化
            if col_name not in not_standard_list:
                mean_value = mean_data.iloc[-1][col_name].item()
                std_value = std_data.iloc[-1][col_name].item()
                data[col_name] = data[col_name] - mean_value
                if std_value != 0:
                    data[col_name] = data[col_name]/std_value
                else:
                    data[col_name] = 0
        data = data.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.drop(columns='symbol', inplace=True)
        data.iloc[:, 1:] = data.iloc[:, 1:].mask((data.iloc[:, 1:] > 1024) | (data.iloc[:, 1:] < -1024), 0)
        financial_past = data.iloc[:, 1:102].drop_duplicates().reset_index(drop=True)
        financial_past = financial_past.iloc[-17:].reset_index(drop=True)
        if len(financial_past) != 17:
            continue
        data.drop(columns='date', inplace=True)
        data = data.values
        data = torch.tensor(data).unsqueeze(0).float().to(device)
        financial_data = financial_past.values

        financial_data = torch.tensor(financial_data).unsqueeze(0).float().to(device)
        with torch.no_grad():  # 禁用梯度计算
            _, label = model(data, financial_data)
        label = label.squeeze(0).cpu().detach().numpy()[0]
        score = label[0]**2/label[1]
        score_idx = {'symbol': [symbol], 'growth': label[0], 'death': label[1], 'score': score}
        score_slice = pd.DataFrame(score_idx)
        score_data = pd.concat([score_data, score_slice], ignore_index=True)
        if score >= 1.0:
            print(symbol + ' score:', score)
    score_data = score_data.sort_values('score', ascending=False)
    score_data = score_data.reset_index(drop=True)

    score_data_name = str(thread_idx) + '.h5'
    if os.path.exists(score_data_name):
        os.remove(score_data_name)
    score_data.to_hdf(score_data_name, key='data', mode='w')


def get_exchange_candidates(exchange):
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('thirteen_points.pt').to(device)
    #处理上交所数据
    upper_exchange = exchange[0].upper() + exchange[1:]
    exchange = exchange[0].lower() + exchange[1:]
    daily_data = pd.read_csv('../data/' + upper_exchange + '/daily_' + exchange + '.csv')
    income_data = pd.read_csv('../data/' + upper_exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv('../data/' + upper_exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv('../data/' + upper_exchange + '/cashflow_' + exchange + '.csv')
    indicator_data = pd.read_csv('../data/' + upper_exchange + '/indicator_' + exchange + '.csv')
    mean_data = pd.read_csv('../data/' + upper_exchange + '/mean_' + exchange + '.csv')
    std_data = pd.read_csv('../data/' + upper_exchange + '/std_' + exchange + '.csv')
    #合并财务相关数据
    financial_data = pd.merge(indicator_data, income_data, on=['symbol', 'endDate'], how='outer')
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

    groups = list(financial_data.groupby('symbol'))
    random.shuffle(groups)
    group_count = len(groups)
    split_count = int(0.25*group_count)
    groups_0 = groups[:split_count]
    groups_1 = groups[split_count:split_count*2]
    groups_2 = groups[split_count*2:split_count*3]
    groups_3 = groups[split_count*3:]

    # 创建线程并启动它们
    thread0 = threading.Thread(target=get_groups_candidates, args=(groups_0, model, daily_data, mean_data, std_data, 0))
    thread1 = threading.Thread(target=get_groups_candidates, args=(groups_1, model, daily_data, mean_data, std_data, 1))
    thread2 = threading.Thread(target=get_groups_candidates, args=(groups_2, model, daily_data, mean_data, std_data, 2))
    thread3 = threading.Thread(target=get_groups_candidates, args=(groups_3, model, daily_data, mean_data, std_data, 3))

    thread0.start()
    thread1.start()
    thread2.start()
    thread3.start()

    thread0.join()
    thread1.join()
    thread2.join()
    thread3.join()

    score_data_0 = pd.read_hdf('0.h5')
    score_data_1 = pd.read_hdf('1.h5')
    score_data_2 = pd.read_hdf('2.h5')
    score_data_3 = pd.read_hdf('3.h5')
    score_data = pd.concat([score_data_0, score_data_1, score_data_2, score_data_3], axis=0)
    score_data = score_data.sort_values('score', ascending=False)
    score_data = score_data.reset_index(drop=True)
    print(score_data)
    score_data.to_csv('../data/' + upper_exchange + '/thirteen_points_candidates.csv', index=False)
    os.remove('0.h5')
    os.remove('1.h5')
    os.remove('2.h5')
    os.remove('3.h5')


def get_candidates():
    get_exchange_candidates('Shanghai')
    get_exchange_candidates('Shenzhen')
    predict_data_shanghai = pd.read_csv('../data/Shanghai/thirteen_points_candidates.csv', engine='pyarrow')
    predict_data_shenzhen = pd.read_csv('../data/Shenzhen/thirteen_points_candidates.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_data.sort_values(by='score', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    predict_data = predict_data.head(29)
    if os.path.exists('candidate_symbols.csv'):
        pre_predict_data = pd.read_csv('candidate_symbols.csv')
        delete_symbols = pre_predict_data[~pre_predict_data['symbol'].isin(predict_data['symbol'])].dropna()
        add_symbols = predict_data[~predict_data['symbol'].isin(pre_predict_data['symbol'])].dropna()
        print('删除的股票：')
        print(delete_symbols)
        print('添加的股票：')
        print(add_symbols)
    predict_data.to_csv('candidate_symbols.csv', index=False)


if __name__ == "__main__":
    get_candidates()
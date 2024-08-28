import os
import torch
from PIL import Image
import numpy as np
from get_stock_data import get_exchange_daily_data, get_exchange_financial_data
import random
import pandas as pd
import datetime
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

not_standard_list = ['open', 'low', 'high', 'close', 'adjClose', 'volume','dividendRatio', 'pbInverse', 'psInverse',
                     'cashFlowInverse', 'debtInverse', 'totalStockholdersEquityInverse', 'quickRatioInverse',
                     'roeInverse', 'interestInverse', 'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayinyear', 'dayoveryear']


def get_data():
    get_exchange_financial_data('Shanghai')
    get_exchange_daily_data('Shanghai')
    get_exchange_financial_data('Shenzhen')
    get_exchange_daily_data('Shenzhen')


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


# def get_groups_candidates(groups, model, daily_data, mean_data, std_data, thread_idx):

def get_exchange_candidates(exchange):
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('twenty_nine_points.pt').to(device)
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
    #得到股数和收盘价的数据和负债和现金数据
    number_of_shares_data = indicator_data[['symbol', 'endDate', 'numberOfShares', 'debtToEquityRatio']]
    close_data = daily_data[['symbol', 'date', 'close']]
    debt_cash_data = balance_data[['symbol', 'endDate', 'totalDebt', 'cashAndCashEquivalents']]
    #得到最新的operatingCashFlow和dividend的均值和方差
    operating_cashflow_mean = mean_data.iloc[-1]['operatingCashFlow']
    operating_cashflow_std = std_data.iloc[-1]['operatingCashFlow']
    dividend_mean = mean_data.iloc[-1]['dividend']
    dividend_std = std_data.iloc[-1]['dividend']
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



    # predict结果
    predict_list = []
    #生成financial_forecast_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        symbol = groups[i][0]
        #得到当前股票的最新股数和收盘价和总负责
        number_of_shares_symbol = number_of_shares_data[number_of_shares_data['symbol'] == symbol].reset_index(drop=True)
        number_of_shares = number_of_shares_symbol.iloc[-1]['numberOfShares']
        debt_cash_symbol = debt_cash_data[debt_cash_data['symbol'] == symbol].reset_index(drop=True)
        debt = debt_cash_symbol.iloc[-1]['totalDebt']
        cash = debt_cash_symbol.iloc[-1]['cashAndCashEquivalents']

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
        if len(daily_group) == 0:
            continue
        #先获得第一天的close和adjClose价格和date
        first_close = daily_group['close'].iloc[0]
        first_adjClose = daily_group['adjClose'].iloc[0]
        first_date = daily_group['date'].iloc[0]

        group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol'])
        group = group.dropna(axis=0, how='any')
        group.reset_index(inplace=True)
        group = group.drop(columns='index')

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
        group = group.reset_index(drop=True)
        group = group.replace([np.inf, -np.inf], np.nan)
        group = group.fillna(0)
        if len(group) < 1907:
            continue
        data = group.iloc[::-1].reset_index(drop=True)
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
        data = data.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.drop(columns='symbol', inplace=True)
        data.drop(columns='date', inplace=True)
        data = data.iloc[:1907].reset_index(drop=True)
        data = data.values
        data = torch.tensor(data).unsqueeze(0).float().to(device)
        with torch.no_grad():  # 禁用梯度计算
            dcf_fore = model(data)
        dcf_fore = dcf_fore.squeeze(0).cpu().detach().numpy()
        ocf_fore = dcf_fore[:, 1]
        dividend_fore = dcf_fore[:, 0]
        ocf_fore = ocf_fore*operating_cashflow_std + operating_cashflow_mean
        dividend_fore = dividend_fore*dividend_std + dividend_mean
        power_fore = np.arange(16, -1, -1)
        ocf_fore = ocf_fore * (0.99 ** power_fore)
        ocf_fore = np.sum(ocf_fore)
        dividend_fore = dividend_fore * (0.99 ** power_fore)
        dividend_fore = np.sum(dividend_fore)
        close_symbol = close_data[close_data['symbol'] == symbol].reset_index(drop=True)
        close = close_symbol.iloc[-1]['close']
        ocf_score = (ocf_fore + cash - debt)/(number_of_shares*close)
        dividend_score = dividend_fore/close
        score = ocf_score + dividend_score
        predict_result = {'symbol': symbol, 'ocf_score': ocf_score, 'dividend_score': dividend_score, 'score':score}
        predict_list.append(predict_result)
        predict_data = pd.DataFrame(predict_list)
        predict_data = predict_data.sort_values('score', ascending=False)
        predict_data = predict_data.reset_index(drop=True)
        print(predict_data.head(7))
    predict_data.to_csv('../data/' + upper_exchange + '/twenty_nine_predict.csv', index=False)

def get_candidates():
    # get_exchange_candidates('Shanghai')
    # get_exchange_candidates('Shenzhen')
    # get_exchange_candidates('HongKong')
    predict_data_shenzhen = pd.read_csv('../data/Shenzhen/thirteen_points_candidates.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv('../data/Shanghai/thirteen_points_candidates.csv', engine='pyarrow')
    predict_data_hongkong = pd.read_csv('../data/HongKong/thirteen_points_candidates.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai, predict_data_hongkong], axis=0)
    predict_data.sort_values(by='score', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    predict_data = predict_data[predict_data['score'] >= 1.63]
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
    # get_exchange_financial_data('Shanghai')
    # get_exchange_daily_data('Shanghai')
    get_exchange_candidates('Shanghai')
    get_exchange_candidates('Shenzhen')

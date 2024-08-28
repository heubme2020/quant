import numpy as np
import pandas as pd
import cupy as cp
import datetime
from tqdm import tqdm
from catboost import CatBoostRegressor


not_standard_list = ['dividendRatio', 'pbInverse', 'debtToEquityRatio', 'quickRatio', 'ebitdaratio',
                     'operatingIncomeRatio', 'incomeBeforeTaxRatio', 'netIncomeRatio', 'grossProfitRatio', 'psInverse',
                     'cashFlowInverse', 'roe', 'dcfInverse', 'open', 'low', 'high', 'close', 'volume', 'marketCap',
                     'dayinyear', 'eps', 'epsdiluted', 'dividend', 'adjClose', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayoveryear']


# 求取上个季度的最后一天
def get_pre_quarter_end_date(date_str):
    date_str = str(date_str)
    date_check = date_str[-4:]
    if date_check == '1231' or date_check == '0331' or date_check == '0630' or date_check == '0930':
        return date_str
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month - 1) // 3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str


def get_exchange_growth_death(exchange):
    print(exchange)
    growth_model = CatBoostRegressor(task_type='GPU')
    growth_model.load_model('catboost_growth_17.bin')
    death_model = CatBoostRegressor(task_type='GPU')
    death_model.load_model('catboost_death_17.bin')
    upper_exchange = exchange
    exchange = exchange.lower()
    income_data = pd.read_csv('../data/' + upper_exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv('../data/' + upper_exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv('../data/' + upper_exchange + '/cashflow_' + exchange + '.csv')
    dividend_data = pd.read_csv('../data/' + upper_exchange + '/dividend_' + exchange + '.csv')
    mean_data = pd.read_csv('../data/' + upper_exchange + '/mean_' + exchange + '.csv')
    std_data = pd.read_csv('../data/' + upper_exchange + '/std_' + exchange + '.csv')

    # 合并财务相关数据
    financial_data = pd.merge(dividend_data, income_data, on=['symbol', 'endDate'], how='outer')
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

    # predict结果
    predict_list = []
    groups = list(financial_data.groupby('symbol'))
    print(len(groups))
    # 生成growth_death_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        group = group.fillna(0).reset_index(drop=True)
        symbol = groups[i][0]
        # 目前必须上市17个季度后才可以预测
        if len(group) < 17:
            continue
        # 下面进行归一化
        input_data = group.iloc[-17:]
        input_data = input_data.reset_index(drop=True)
        input_data = input_data.drop('symbol', axis=1)
        input_data = input_data.drop('endDate', axis=1)
        input_data = input_data.astype('float32')
        col_names = input_data.columns.values
        for k in range(len(input_data.columns)):
            col_name = col_names[k]
            if col_name in not_standard_list:
                continue
            mean_value = mean_data[col_name].iloc[-1]
            std_value = std_data[col_name].iloc[-1]
            # sigma_max = mean_value + 17 * std_value
            # sigma_min = mean_value - 17 * std_value
            # input_data.loc[input_data[col_name] > sigma_max, col_name] = np.float32(sigma_max)
            # input_data.loc[input_data[col_name] < sigma_min, col_name] = np.float32(sigma_max)
            input_data[col_name] = input_data[col_name] - mean_value
            if std_value != 0:
                input_data[col_name] = input_data[col_name] / std_value
            else:
                input_data[col_name] = 0
        input_data = input_data.fillna(0)
        input_data.replace([np.inf, -np.inf], 0, inplace=True)
        input_data = input_data.apply(lambda x: 0 if isinstance(x, (int, float)) and (
                x > 1204 or x < -1024 or (-1.0 / 1024 < x < 1.0 / 1024)) else x)
        input_data = input_data.T
        input_data = np.ravel(input_data.values)
        input_data = pd.DataFrame(input_data)
        input_data = cp.asnumpy(input_data.T)
        # input_data = np.asnumpy(input_data.T)
        predict_growth = growth_model.predict(input_data)
        predict_growth = predict_growth[0]
        predict_death = death_model.predict(input_data)
        predict_death = predict_death[0]
        predict_result = {'symbol': symbol, 'growth': predict_growth, 'death': predict_death}
        predict_list.append(predict_result)
    predict_data = pd.DataFrame(predict_list)
    predict_data['rank'] = predict_data['growth'] / predict_data['death']
    predict_data = predict_data.sort_values('rank', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    predict_data.to_csv('../data/' + upper_exchange + '/grow_death_predict.csv', index=False)


def refresh_growth_death():
    get_exchange_growth_death('Shenzhen')
    get_exchange_growth_death('Shanghai')
    # get_exchange_growth_death('HongKong')
    predict_data_shenzhen = pd.read_csv('../data/Shenzhen/grow_death_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv('../data/Shanghai/grow_death_predict.csv', engine='pyarrow')
    # predict_data_hongkong = pd.read_csv('../data/HongKong/grow_death_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_data.sort_values(by='rank', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    predict_data = predict_data.head(43)
    pre_predict_data = pd.read_csv('candidate_symbols.csv')
    delete_symbols = pre_predict_data[~pre_predict_data['symbol'].isin(predict_data['symbol'])].dropna()
    add_symbols = predict_data[~predict_data['symbol'].isin(pre_predict_data['symbol'])].dropna()
    print('删除的股票：')
    print(delete_symbols)
    print('添加的股票：')
    print(add_symbols)
    predict_data.to_csv('candidate_symbols.csv', index=False)


if __name__ == '__main__':
    refresh_growth_death()


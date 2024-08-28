import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
from self_attention_model import Growth_Death
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from datetime import datetime, timedelta
import torch.optim as optim
import mmap

e_power = 1.0/math.e

# 选择要保留的列
columns_to_keep = ['date', 'open', 'low', 'high', 'close', 'adjClose', 'volume', 'delta', 'dayinyear', 'dayoveryear']
not_standard_list = ['open', 'low', 'high', 'close', 'adjClose', 'volume','dividendRatio', 'pbInverse', 'psInverse',
                     'cashFlowInverse', 'debtInverse', 'totalStockholdersEquityInverse', 'quickRatioInverse',
                     'roeInverse', 'interestInverse', 'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayinyear', 'dayoveryear']


def get_quarter_first_day(date):
    # 将输入的日期字符串转换为日期对象
    date_obj = datetime.strptime(str(int(date)), '%Y%m%d')
    # 获取日期所在季度的第一个月份
    quarter_month = ((date_obj.month - 1) // 3) * 3 + 1
    # 获取日期所在季度的年份
    quarter_year = date_obj.year
    # 计算本季度第一天日期
    this_quarter_first_day = datetime(quarter_year, quarter_month, 1)

    # 计算上个季度第一天日期
    last_quarter_month = quarter_month - 3 if quarter_month > 3 else 9 + quarter_month
    last_quarter_year = quarter_year - 1 if quarter_month == 1 else quarter_year
    last_quarter_first_day = datetime(last_quarter_year, last_quarter_month, 1)

    # 计算上上个季度第一天日期
    last_last_quarter_month = last_quarter_month - 3 if last_quarter_month > 3 else 9 + last_quarter_month
    last_last_quarter_year = last_quarter_year - 1 if last_quarter_month == 1 else last_quarter_year
    last_last_quarter_first_day = datetime(last_last_quarter_year, last_last_quarter_month, 1)

    return float(this_quarter_first_day.strftime('%Y%m%d')), \
           float(last_quarter_first_day.strftime('%Y%m%d')),\
           float(last_last_quarter_first_day.strftime('%Y%m%d'))


def fix_date(date_float):
    date_str = str(int(date_float))
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    if month < 1:
        month = 1
    elif month > 12:
        month = 12
    if day < 1:
        day = 1
    last_day_of_month = pd.Timestamp(year, month, 1).days_in_month
    if day > last_day_of_month:
        day = last_day_of_month

    fixed_date_str = f"{year:04d}{month:02d}{day:02d}"
    return float(fixed_date_str)

class Dataset(Dataset):
    def __init__(self, bin_file, index_file):
        self.bin_file = bin_file
        with open(index_file, 'r') as f:
            self.index = [line.strip().split('\t') for line in f]
        self.columns = pd.read_csv('columns.csv', header=None).squeeze().tolist()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        offset, num_rows, num_cols = map(int, self.index[idx])
        byte_offset = offset
        byte_length = num_rows * num_cols * 4  # float32 has 4 bytes

        with open(self.bin_file, 'rb') as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            mmapped_file.seek(byte_offset)
            data = np.frombuffer(mmapped_file.read(byte_length), dtype=np.float32)
            mmapped_file.close()
        data = data.reshape((num_rows, num_cols))
        data = pd.DataFrame(data, columns=self.columns)
        data = data.replace([np.inf, -np.inf], 0)
        data = data.fillna(0).reset_index(drop=True)
        data.iloc[:, 1:] = data.iloc[:, 1:].mask((data.iloc[:, 1:] > 1024) | (data.iloc[:, 1:] < -1024), 0)
        data_past, financial_past, financial_fore, growth, death = get_data_input(data)
        date = data_past.iloc[-1]['date']
        date = fix_date(date)

        #修改data_past按概率消掉前三个季度财务的信息
        quater_first_day, last_quarter_first_day, last_last_quarter_first_day = get_quarter_first_day(date)
        if np.random.choice([True, False]):
            rows_to_modify = data_past[data_past['date'] > quater_first_day]
            data_past.loc[rows_to_modify.index, ~data_past.columns.isin(columns_to_keep)] = 0
            if np.random.choice([True, False]):
                rows_to_modify = data_past[data_past['date'] > last_quarter_first_day]
                data_past.loc[rows_to_modify.index, ~data_past.columns.isin(columns_to_keep)] = 0
                if np.random.choice([True, False]):
                    rows_to_modify = data_past[data_past['date'] > last_last_quarter_first_day]
                    data_past.loc[rows_to_modify.index, ~data_past.columns.isin(columns_to_keep)] = 0
        data_past = data_past.drop('date', axis=1)
        data_past = data_past.values
        data_past = torch.tensor(data_past).float()

        # 修改financial_past按概率消掉前三个季度财务的信息
        if np.random.choice([True, False]):
            financial_past.iloc[-1] = 0
            if np.random.choice([True, False]):
                financial_past.iloc[-2] = 0
                if np.random.choice([True, False]):
                    financial_past.iloc[-3] = 0

        financial_past = financial_past.values
        financial_past = torch.tensor(financial_past).float()
        financial_fore = financial_fore.values
        financial_fore = torch.tensor(financial_fore).float()
        label = pd.DataFrame({'growth': [growth], 'death': [death]})
        label = label.values
        label = torch.tensor(label).float()
        # print('data_past:', data_past)
        # print('financial_past:', financial_past)
        return data_past.to('cuda'), financial_past.to('cuda'), financial_fore.to('cuda'), label.to('cuda')


def collate_fn(batch):
    filtered_batch = []
    for item in batch:
        tensor0, tensor1, tensor2, tensor3 = item
        if tensor1.shape[0] == 17 and tensor2.shape[0] == 7:
            filtered_batch.append(item)
    if len(filtered_batch) == 0:
        return None  # 返回 None 或一个空 Tensor，取决于具体需求
    tensors0, tensors1, tensors2, tensors3 = zip(*filtered_batch)
    return torch.stack(tensors0), torch.stack(tensors1), torch.stack(tensors2), torch.stack(tensors3)


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


def get_quarter_first_day(date):
    # 将输入的日期字符串转换为日期对象
    date_obj = datetime.strptime(str(int(date)), '%Y%m%d')
    # 获取日期所在季度的第一个月份
    quarter_month = ((date_obj.month - 1) // 3) * 3 + 1
    # 获取日期所在季度的年份
    quarter_year = date_obj.year
    # 计算本季度第一天日期
    this_quarter_first_day = datetime(quarter_year, quarter_month, 1)

    # 计算上个季度第一天日期
    last_quarter_month = quarter_month - 3 if quarter_month > 3 else 9 + quarter_month
    last_quarter_year = quarter_year - 1 if quarter_month == 1 else quarter_year
    last_quarter_first_day = datetime(last_quarter_year, last_quarter_month, 1)

    # 计算上上个季度第一天日期
    last_last_quarter_month = last_quarter_month - 3 if last_quarter_month > 3 else 9 + last_quarter_month
    last_last_quarter_year = last_quarter_year - 1 if last_quarter_month == 1 else last_quarter_year
    last_last_quarter_first_day = datetime(last_last_quarter_year, last_last_quarter_month, 1)

    return float(this_quarter_first_day.strftime('%Y%m%d')), \
           float(last_quarter_first_day.strftime('%Y%m%d')),\
           float(last_last_quarter_first_day.strftime('%Y%m%d'))


def get_ttm_tensor(tensor_data):
    # 计算每行的均值
    mean_0 = torch.mean(tensor_data[:, 0:1, :], dim=1)
    mean_1 = torch.mean(tensor_data[:, 0:2, :], dim=1)
    mean_2 = torch.mean(tensor_data[:, 0:3, :], dim=1)
    mean_3 = torch.mean(tensor_data[:, 0:5, :], dim=1)
    mean_4 = torch.mean(tensor_data[:, 0:7, :], dim=1)
    # mean_5 = torch.mean(tensor_data[:, 0:6, :], dim=1)

    # 构造新的张量
    mean_tensor = torch.zeros_like(tensor_data).to(tensor_data.device)
    mean_tensor[:, 0, :] = mean_0
    mean_tensor[:, 1, :] = mean_1
    mean_tensor[:, 2, :] = mean_2
    mean_tensor[:, 3, :] = mean_3
    mean_tensor[:, 4, :] = mean_4
    # mean_tensor[:, 5, :] = mean_5
    return mean_tensor


def train_thirteen_model(model, model_name, batch_size):
    device = 'cuda'
    train_dataset = Dataset('train_data.bin', 'train_index.txt')
    val_dataset = Dataset('train_data.bin', 'validate_index.txt')

    if os.path.exists(model_name):
        model = torch.load(model_name).to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()
    learning_rate = 0.001
    num_epochs = 43
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5,
                                  persistent_workers=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5,
                                persistent_workers=True, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5,
    #                               persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5,
    #                             persistent_workers=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    #读取feature_importance.csv，作为计算预测财务数据loss时权重
    feature_importance = pd.read_csv('../three_points/feature_importance.csv')
    weight_values = feature_importance.values[0]
    weights = torch.tensor(weight_values).unsqueeze(0).unsqueeze(0).to('cuda')
    # print(weight_values)
    for epoch in range(num_epochs):
        # 训练阶段
        mean_train_loss = 0.0
        step_num = 0
        for batch in train_dataloader:
            if batch is None:
                continue  # 跳过无效的批次
            data_past, financial_past, financial_fore, label = batch
            optimizer.zero_grad()
            financial_predict, label_predict = model(data_past, financial_past)
            # # 检查 NaN
            # if torch.isnan(data_past).any() or torch.isnan(financial_past).any() or torch.isnan(financial_predict).any() or label_predict:
            #     continue  # 跳过包含 NaN 的样本
            financial_predict_ttm = get_ttm_tensor(financial_predict)
            financial_fore_ttm = get_ttm_tensor(financial_fore)
            loss_financial = torch.mean(torch.abs(financial_predict_ttm - financial_fore_ttm) * weights)
            loss_growth_death = criterion(label_predict, label)
            # print('loss_financial:', loss_financial.item())
            # print('loss_growth_death:', loss_growth_death.item())
            loss = loss_financial + loss_growth_death
            loss.backward()  # 计算梯度
            optimizer.step()
            mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
            step_num = step_num + 1
            print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min val loss: %1.5f" % (epoch, loss.item(),
                                                                                           mean_train_loss,
                                                                                           best_val_loss))

        # 验证阶段
        mean_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue  # 跳过无效的批次
                data_past, financial_past, financial_fore, label = batch
                financial_predict, label_predict = model(data_past, financial_past)
                # if torch.isnan(data_past).any() or torch.isnan(financial_past).any() or torch.isnan(
                #         financial_predict).any() or label_predict:
                #     continue  # 跳过包含 NaN 的样本
                financial_predict_ttm = get_ttm_tensor(financial_predict)
                financial_fore_ttm = get_ttm_tensor(financial_fore)
                # print(financial_fore_ttm.shape)
                loss_financial = torch.mean(torch.abs(financial_predict_ttm - financial_fore_ttm) * weights)
                # print(loss_financial.item())
                loss_growth_death = criterion(label_predict, label)
                loss = loss_financial + loss_growth_death
                mean_val_loss += loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)


if __name__ == '__main__':
    model_name = 'thirteen_points.pt'
    model = Growth_Death([123, 1201], [101, 7]).to('cuda')
    batch_size = 32
    train_thirteen_model(model, model_name, batch_size)


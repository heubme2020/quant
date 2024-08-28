import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
from twenty_nine_model import DCF
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from datetime import datetime, timedelta
import torch.optim as optim
import mmap

# 选择要保留的列
columns_to_keep = ['date', 'open', 'low', 'high', 'close', 'adjClose', 'volume', 'delta', 'dayinyear', 'dayoveryear']


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
        data_past, financial_fore = get_data_input(data)
        data_past = data_past.values
        data_past = torch.tensor(data_past).float()
        dividend_fore = financial_fore['dividend'].values
        dividend_fore = torch.tensor(dividend_fore).unsqueeze(1).float()
        ocf_fore = financial_fore['operatingCashFlow'].values
        ocf_fore = torch.tensor(ocf_fore).unsqueeze(1).float()
        dcf_fore = torch.cat((dividend_fore, ocf_fore), dim=1)
        return data_past.to('cuda'), dcf_fore.to('cuda')


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


def get_data_input(data):
    data = data.iloc[::-1].reset_index(drop=True)
    data_past = data.iloc[1093:].reset_index(drop=True)
    date_last = data_past.iloc[0]['date']
    date_last = fix_date(date_last)
    #随机将临近季度财务数据置0
    quater_first_day, last_quarter_first_day, last_last_quarter_first_day = get_quarter_first_day(date_last)
    if np.random.choice([True, False]):
        rows_to_modify = data_past[data_past['date'] > quater_first_day]
        data_past.loc[rows_to_modify.index, ~data_past.columns.isin(columns_to_keep)] = 0
        if np.random.choice([True, False]):
            rows_to_modify = data_past[data_past['date'] > last_quarter_first_day]
            data_past.loc[rows_to_modify.index, ~data_past.columns.isin(columns_to_keep)] = 0
            if np.random.choice([True, False]):
                rows_to_modify = data_past[data_past['date'] > last_last_quarter_first_day]
                # 将除了指定列之外的值设为 0
                data_past.loc[rows_to_modify.index, ~data_past.columns.isin(columns_to_keep)] = 0
    data_past = data_past.drop('date', axis=1)

    data_fore = data.iloc[:1093]
    financial_fore = data_fore.iloc[:, 1:102].drop_duplicates().reset_index(drop=True)
    season_diff = 17 - len(financial_fore)
    if season_diff > 0:
        financial_fore = pd.concat([financial_fore]*17, axis=0).reset_index(drop=True)
    financial_fore = financial_fore.iloc[:17]
    # financial_fore = financial_fore.reset_index(drop=True)

    return data_past, financial_fore


def get_quarter_first_day(date):
    # print(date)
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


def train_twenty_three_model(model, model_name, batch_size):
    device = 'cuda'
    train_dataset = Dataset('train_data.bin', 'train_index.txt')
    val_dataset = Dataset('train_data.bin', 'validate_index.txt')

    if os.path.exists(model_name):
        model = torch.load(model_name)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    learning_rate = 0.001
    num_epochs = 200
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                persistent_workers=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        mean_train_loss = 0.0
        step_num = 0
        for data_past, dcf_fore in train_dataloader:
            optimizer.zero_grad()
            # print(data_past.shape)
            # print(dcf_fore.shape)
            dcf_predict = model(data_past)
            # dcf_predict = dcf_predict.squeeze(2)
            loss = criterion(dcf_predict[:, -1:], dcf_fore[:, -1:]) + \
                criterion(dcf_predict[:, -2:].mean(dim=1), dcf_fore[:, -2:].mean(dim=1)) + \
                criterion(dcf_predict[:, -3:].mean(dim=1), dcf_fore[:, -3:].mean(dim=1)) + \
                criterion(dcf_predict[:, -5:].mean(dim=1), dcf_fore[:, -5:].mean(dim=1)) + \
                criterion(dcf_predict[:, -7:].mean(dim=1), dcf_fore[:, -7:].mean(dim=1)) + \
                criterion(dcf_predict[:, -11:].mean(dim=1), dcf_fore[:, -11:].mean(dim=1)) + \
                criterion(dcf_predict[:, -13:].mean(dim=1), dcf_fore[:, -13:].mean(dim=1)) + \
                criterion(dcf_predict[:, -17:].mean(dim=1), dcf_fore[:, -17:].mean(dim=1))

            loss.backward()  # 计算梯度
            # optimizer.step()
            optimizer.step()
            # running_train_loss += loss.item() * inputs.size(0)
            mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
            step_num = step_num + 1
            print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min val loss: %1.5f" % (epoch, loss.item(),
                                                                                           mean_train_loss,
                                                                                           best_val_loss))

        # 验证阶段
        mean_val_loss = 0
        with torch.no_grad():
            for data_past, dcf_fore, in val_dataloader:
                dcf_predict = model(data_past)
                # dcf_predict = dcf_predict.squeeze(2)
                loss = criterion(dcf_predict[:, -1:], dcf_fore[:, -1:]) + \
                    criterion(dcf_predict[:, -2:].mean(dim=1), dcf_fore[:, -2:].mean(dim=1)) + \
                    criterion(dcf_predict[:, -3:].mean(dim=1), dcf_fore[:, -3:].mean(dim=1)) + \
                    criterion(dcf_predict[:, -5:].mean(dim=1), dcf_fore[:, -5:].mean(dim=1)) + \
                    criterion(dcf_predict[:, -7:].mean(dim=1), dcf_fore[:, -7:].mean(dim=1)) + \
                    criterion(dcf_predict[:, -11:].mean(dim=1), dcf_fore[:, -11:].mean(dim=1)) + \
                    criterion(dcf_predict[:, -13:].mean(dim=1), dcf_fore[:, -13:].mean(dim=1)) + \
                    criterion(dcf_predict[:, -17:].mean(dim=1), dcf_fore[:, -17:].mean(dim=1))
                mean_val_loss += loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)



if __name__ == '__main__':
    model_name = 'twenty_nine_points.pt'
    model = DCF([123, 1907], [2, 17]).to('cuda')
    batch_size = 16
    train_twenty_three_model(model, model_name, batch_size)

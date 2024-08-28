import datetime
import os
import pandas as pd
from tqdm import tqdm
import random
from dateutil.relativedelta import relativedelta
import numpy as np
import threading


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


if __name__ == '__main__':
    convert_data()
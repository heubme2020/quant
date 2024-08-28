import numpy as np
import pandas as pd
import os

from catboost import CatBoostRegressor, Pool, cv


def train_catboost():
    train_data = pd.read_csv('growth_death_train_data_17.csv')
    # print(len(train_data))
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    # train_data = train_data.dropna()
    # train_data = train_data.reset_index(drop=True)
    print(len(train_data))

    train_num = len(train_data)
    ratio = 0.3
    train_num = int((1.0 - ratio) * train_num)
    train_input = train_data.iloc[:train_num, 2:-2]
    train_growth = train_data.iloc[:train_num, -2]
    data_num = len(train_input)

    validate_input = train_data.iloc[train_num:, 2:-2].reset_index(drop=True)
    validate_growth = train_data.iloc[train_num:, -2].reset_index(drop=True)
    model_growth = CatBoostRegressor(iterations=data_num, learning_rate=0.01, min_data_in_leaf=23, depth=7,
                                     loss_function='MAPE', eval_metric='MAPE', task_type='GPU', devices='0:1',
                                     early_stopping_rounds=2048)
    # 训练模型，并设置验证集
    model_growth.fit(
        train_input, train_growth,
        eval_set=(validate_input, validate_growth),
        verbose=1
    )
    model_growth.save_model('catboost_growth_17.bin')

    # 训练死亡模型
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    # train_data = train_data.dropna()
    # train_data = train_data.reset_index(drop=True)
    print(len(train_data))

    train_num = len(train_data)
    ratio = 0.3
    train_num = int((1.0 - ratio) * train_num)
    train_input = train_data.iloc[:train_num, 2:-2]
    train_death = train_data.iloc[:train_num, -1]
    data_num = len(train_input)

    validate_input = train_data.iloc[train_num:, 2:-2].reset_index(drop=True)
    validate_death = train_data.iloc[train_num:, -1].reset_index(drop=True)
    model_death = CatBoostRegressor(iterations=data_num, learning_rate=0.01, min_data_in_leaf=23, depth=7,
                                    loss_function='MAPE', eval_metric='MAPE', task_type='GPU', devices='0:1',
                                    early_stopping_rounds=2048)
    # 训练模型，并设置验证集
    model_death.fit(
        train_input, train_death,
        eval_set=(validate_input, validate_death),
        verbose=1
    )
    model_death.save_model('catboost_death_17.bin')


def get_catboost_feature_importance():
    growth_model = CatBoostRegressor()
    growth_model.load_model('catboost_growth_17.bin')
    # 获取特征重要性得分
    growth_feature_importance = growth_model.get_feature_importance()
    growth_feature_importance = np.array(growth_feature_importance).reshape(101, 17)

    death_model = CatBoostRegressor()
    death_model.load_model('catboost_death_17.bin')
    # 获取特征重要性得分
    death_feature_importance = death_model.get_feature_importance()
    death_feature_importance = np.array(death_feature_importance).reshape(101, 17)

    feature_importance = growth_feature_importance + death_feature_importance
    feature_importance = feature_importance.T
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance = feature_importance.sum()
    feature_importance = pd.DataFrame(feature_importance)
    # 删除其中的一部分
    feature_importance = feature_importance.T
    feature_data = pd.read_csv('feature_importance.csv')
    os.remove('feature_importance.csv')
    print(feature_data)
    feature_importance.columns = feature_data.columns
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(feature_importance)


def do_catboost():
    train_catboost()
    get_catboost_feature_importance()


if __name__ == '__main__':
    do_catboost()


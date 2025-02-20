import numpy as np
import random
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from collections import defaultdict


def date_deal(test_list):
    year_price_list = [
        [int(row[0]) if row[0] != 'false' else 0,
         int(row[5]) if row[5] != 'false' else 0,
         int(row[-1]) if row[-1] != 'false' else 0]
        for row in test_list
    ]
    return year_price_list


def select_samples(year_price_list):
    year_dict = defaultdict(list)

    # 按年份分类房源
    for item in year_price_list:
        house_id, year, price_per_sqm = item
        year_dict[year].append(item)

    # 选取每个年份的一个样本
    sample_set = [random.choice(houses) for houses in year_dict.values()]

    # 随机填补样本集到 200 个
    all_samples = sum(year_dict.values(), [])  # 展开所有房源
    while len(sample_set) < 200:
        sample_set.append(random.choice(all_samples))

    return sample_set


def log_func(x, a, b):
    return a * np.log(x) + b


def fit_log_relationship(samples):
    years = np.array([item[1] for item in samples])
    prices = np.array([item[2] for item in samples])

    params, _ = opt.curve_fit(log_func, years, prices)
    return params


def adjust_model(year_price_list, params):
    years = np.array([item[1] for item in year_price_list]).reshape(-1, 1)
    prices = np.array([item[2] for item in year_price_list])

    # 计算对数特征
    log_years = np.log(years)

    # 线性回归微调模型
    model = LinearRegression()
    model.fit(log_years, prices)
    slope = round(float(model.coef_[0]), 4)
    point = round(float(model.intercept_), 4)

    return [slope, point]


def year_to_price_return(test_list):
    year_price_list = date_deal(test_list)
    selected_samples = select_samples(year_price_list)
    params = fit_log_relationship(selected_samples)
    adjusted_params = adjust_model(year_price_list, params)
    return adjusted_params


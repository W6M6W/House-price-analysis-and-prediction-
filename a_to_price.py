import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def data_deal(no_clean_list):
    return [
        [int(row[0]) if row[0] != 'false' else 0, float(row[2]) if row[2] != 'false' else 0.0,
         int(row[-1]) if row[-1] != 'false' else 0]
        for row in no_clean_list
    ]


def or_area_price_choice(area_price_list, random_size):
    random_sample = random.choices(area_price_list, k=random_size)
    filtered_data = [row for row in random_sample if not any(x == 0 or x == 0.0 for x in row)]
    return filtered_data


def new_area_price_choice(old_ge, random_size):
    random_sample = random.choices(old_ge, k=random_size)
    filtered_data = [row for row in random_sample if not any(x == 0 or x == 0.0 for x in row)]
    return filtered_data


def area_price_cacul(filtered_data):
    if not filtered_data:
        return None  # 防止空数据导致错误

    index = [row[0] for row in filtered_data]
    max_degree = 4
    X = np.array([x[1] for x in filtered_data]).reshape(-1, 1)
    Y = np.array([x[2] for x in filtered_data])

    best_degree = 1
    best_r2 = -float('inf')
    best_model = None
    best_slope = []

    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, Y)
        Y_pred = model.predict(X_poly)
        r2 = r2_score(Y, Y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_degree = degree
            best_model = model
            best_slope = [round(float(coef), 4) for coef in model.coef_]

    poly = PolynomialFeatures(degree=best_degree)
    X_poly = poly.fit_transform(X)
    Y_pred = best_model.predict(X_poly)
    intercept = round(float(best_model.intercept_), 8)

    return [filtered_data, best_slope, intercept, best_r2]


def a_p_evolution_process(old_ge, random_size, evolut_time, remain_size, new_ge_size, area_price_list):
    total_sample_list = []
    for _ in range(evolut_time):
        total_sample_list.append(area_price_cacul(random.sample(old_ge, random_size)))

    test_size = int((random_size * evolut_time) / 4)
    test_list = area_price_cacul(or_area_price_choice(area_price_list, test_size))

    def to_vector(data):
        if data is None:
            return np.zeros(5)  # 处理空数据
        best_slope, intercept, best_r2 = data[1], data[2], data[3]
        return np.concatenate([np.array(best_slope), np.array([intercept, best_r2])])

    total_vectors = [to_vector(arr) for arr in total_sample_list if arr is not None]
    test_vector = to_vector(test_list)

    distances = [np.linalg.norm(vec - test_vector) for vec in total_vectors]
    best_match_indices = np.argsort(distances)[:remain_size]

    remain_ge = [total_sample_list[i][0] for i in best_match_indices if total_sample_list[i] is not None]
    remain_ge = sum(remain_ge, [])  # 合并多个列表
    remain_ge.extend(or_area_price_choice(area_price_list, new_ge_size))

    return remain_ge


def a_p_evolution(no_clean_list):
    area_price_list = data_deal(no_clean_list)
    first_ge = or_area_price_choice(area_price_list, 1000)
    sec_ge_b = a_p_evolution_process(first_ge, 100, 100, 2000, 2600, area_price_list)
    sec_ge_t = new_area_price_choice(sec_ge_b, 4000)
    third_ge = a_p_evolution_process(sec_ge_t, 4000, 10, 1000, 1600, area_price_list)
    final_answer = area_price_cacul(third_ge)

    if final_answer is None:
        return None

    f_slope = final_answer[1]
    f_intercept = final_answer[2]
    f_r2 = final_answer[3]

    return [f_slope, f_intercept, f_r2]


















import numpy as np

def data_deal(test_list):
    le_price_list = [
        [int(row[0]) if row[0] != 'false' else 0, row[4], int(row[-1]) if row[-1] != 'false' else 0]
        for row in test_list
    ]
    return le_price_list


def le_to_price_return(test_list):
    test_list = data_deal(test_list)

    high_floor_data, low_floor_data = [], []
    high_floor_high, high_floor_mid, high_floor_low = [], [], []
    low_floor_high, low_floor_mid, low_floor_low = [], [], []

    for entry in test_list:
        floor_info = entry[1]

        # 提取楼层总数
        try:
            total_floors = int(floor_info.split('共')[1].split('层')[0])
        except (IndexError, ValueError):
            continue  # 跳过格式错误的数据

        # 根据楼层总数分类
        if total_floors > 8:
            high_floor_data.append(entry)
            if '高层' in floor_info:
                high_floor_high.append(entry)
            elif '中层' in floor_info:
                high_floor_mid.append(entry)
            else:
                high_floor_low.append(entry)
        else:
            low_floor_data.append(entry)
            if '高层' in floor_info:
                low_floor_high.append(entry)
            elif '中层' in floor_info:
                low_floor_mid.append(entry)
            else:
                low_floor_low.append(entry)

    # 计算总平均房价
    all_unit_prices = [entry[2] for entry in test_list]
    total_avg_price = np.mean(all_unit_prices) if all_unit_prices else 0

    def safe_avg(data):
        return sum(row[2] for row in data) / len(data) if data else total_avg_price

    # 计算不同楼层类型的平均房价
    high_floor_avg_price = safe_avg(high_floor_data)
    low_floor_avg_price = safe_avg(low_floor_data)
    high_floor_high_avg_price = safe_avg(high_floor_high)
    high_floor_mid_avg_price = safe_avg(high_floor_mid)
    high_floor_low_avg_price = safe_avg(high_floor_low)
    low_floor_high_avg_price = safe_avg(low_floor_high)
    low_floor_mid_avg_price = safe_avg(low_floor_mid)
    low_floor_low_avg_price = safe_avg(low_floor_low)

    # 计算不同楼层价格与平均价格的百分比差异
    def percent_diff(value, base):
        return ((value - base) / base)  if base else 0

    return [
        percent_diff(high_floor_avg_price, total_avg_price),
        percent_diff(low_floor_avg_price, total_avg_price),
        percent_diff(high_floor_high_avg_price, high_floor_avg_price),
        percent_diff(high_floor_mid_avg_price, high_floor_avg_price),
        percent_diff(high_floor_low_avg_price, high_floor_avg_price),
        percent_diff(low_floor_high_avg_price, high_floor_avg_price),
        percent_diff(low_floor_mid_avg_price, high_floor_avg_price),
        percent_diff(low_floor_low_avg_price, high_floor_avg_price)
    ]





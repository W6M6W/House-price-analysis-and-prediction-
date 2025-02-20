def data_clean(no_clean_list):
    ad_price_list = []
    for row in no_clean_list:
        try:
            ad_id = int(row[0]) if row[0] != 'false' else 0  # 数字化ID
            region_code = get_region_code(row[6])  # 获取地区数字化标记
            price = int(row[-1]) if row[-1] != 'false' else 0  # 数字化价格
            if region_code != -1:  # 过滤无效地区数据
                ad_price_list.append([ad_id, region_code, price])
        except (ValueError, IndexError):
            continue  # 遇到异常数据时跳过
    return ad_price_list


def get_region_code(address):
    region_map = {
        '蒸湘': 0, '华新': 1, '石鼓': 2, '珠晖': 3,
        '立新': 4, '雁峰': 5, '南岳': 6
    }
    for key, value in region_map.items():
        if key in address:
            return value
    return -1  # 其他地区用-1表示


def ad_to_price_return(no_deal_list):
    test_list = data_clean(no_deal_list)
    if not test_list:
        return [0] * 8  # 避免空数据导致错误

    total_price = sum(row[2] for row in test_list)
    total_aver_price = total_price / len(test_list) if test_list else 0

    area_price_list = [[] for _ in range(8)]
    for _, region_code, price in test_list:
        if 0 <= region_code < 7:  # 确保索引不超出范围
            area_price_list[region_code].append(price)

    di_area_price_per = []
    for prices in area_price_list:
        if prices:
            avg_price = sum(prices) / len(prices)
            di_area_price_per.append((avg_price - total_aver_price) / total_aver_price)
        else:
            di_area_price_per.append(0)  # 避免空列表导致计算错误

    return di_area_price_per

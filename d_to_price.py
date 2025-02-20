
def data_deal(no_clean_list):
    drice_price_list = [
        [int(row[0]) if row[0] != 'false' else 0,  # 编号
         1 if row[3] in ['南北', '南'] else 0,     # 朝向 (南 or 南北为1，其他为0)
         int(row[-1]) if row[-1] != 'false' else 0]  # 每平米单价
        for row in no_clean_list
    ]
    return drice_price_list


def drice_price_return(test_list):
    drice_price_list = data_deal(test_list)

    # 根据朝向拆分数据
    drice_sn = [row for row in drice_price_list if row[1] == 1]
    dice_other = [row for row in drice_price_list if row[1] == 0]

    # 避免除零错误
    whole_aver_price = sum(row[2] for row in drice_price_list) / len(drice_price_list) if drice_price_list else 0
    sn_aver_price = sum(row[2] for row in drice_sn) / len(drice_sn) if drice_sn else whole_aver_price
    other_aver_price = sum(row[2] for row in dice_other) / len(dice_other) if dice_other else whole_aver_price

    # 计算百分比变化
    percent_increase_sn = ((sn_aver_price - whole_aver_price) / whole_aver_price) * 100 if whole_aver_price else 0
    percent_decrease_other = ((other_aver_price - whole_aver_price) / whole_aver_price) * 100 if whole_aver_price else 0

    return [percent_increase_sn, percent_decrease_other]


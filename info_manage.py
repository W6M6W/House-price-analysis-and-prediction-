import re

def remove_duplicates(input_file):
    """
    读取数据文件，去除重复房源信息（忽略编号）。
    """
    unique_properties = set()
    cleaned_lines = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            match = re.match(r'编号：\d+，(房产名称：“[^”]+”.*)', line.strip())
            if match:
                property_info = match.group(1)

                if property_info not in unique_properties:
                    unique_properties.add(property_info)
                    cleaned_lines.append(property_info)

    return cleaned_lines


def parse_and_format(cleaned_lines):
    """
    解析并转换数据格式，确保所有字段完整后再重新编号。
    """
    formatted_data = []
    for property_info in cleaned_lines:
        match = re.match(
            r'房产名称：“([^”]+)”，格局：(\d+)室(\d+)厅(\d+)卫，面积：([0-9.]+)㎡，朝向：([^，]+)，楼层：([^，]+)，建造年代：(\d+)，地址：([^，]+)，总价：([0-9.]+)万，单价：([0-9.]+)元/㎡',
            property_info)

        if match:
            rooms = match.group(2)   # 室
            halls = match.group(3)   # 厅
            baths = match.group(4)   # 卫
            area = match.group(5)    # 面积
            direction = match.group(6)  # 朝向
            floor = match.group(7)   # 楼层
            year = match.group(8)    # 建造年代
            address = match.group(9)  # 地址
            total_price = float(match.group(10)) * 10000  # 总价（转换为元）
            unit_price = match.group(11)  # 单价

            # 格局格式化为 "422"（代表 4 室 2 厅 2 卫）
            layout = f"{rooms}{halls}{baths}"

            # 确保单价是整数
            unit_price = int(float(unit_price))

            formatted_data.append(f"{layout} {area} {direction} {floor} {year} {address} {int(total_price)} {unit_price}")

    return formatted_data


def reindex_and_save(formatted_data, output_file):
    """
    解析数据后再进行编号，确保编号连续，并输出符合指定格式。
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for index, property_info in enumerate(formatted_data, start=1):
            updated_line = f"{index} {property_info}"
            outfile.write(updated_line + "\n")


def clean_data(input_file, output_file):
    """
    清理数据：去重、解析并转换格式、最后重新编号。
    """
    # 1. 去重
    cleaned_lines = remove_duplicates(input_file)
    print(f"去重完成，剩余 {len(cleaned_lines)} 条房源数据。")

    # 2. 解析并转换数据格式
    formatted_data = parse_and_format(cleaned_lines)
    print(f"数据解析完成，成功解析 {len(formatted_data)} 条房源信息。")

    # 3. 重新编号并保存
    reindex_and_save(formatted_data, output_file)
    print(f"数据清理完成，格式化后的文件已保存至 '{output_file}'")

if __name__ == "__main__":
    input_file = 'total_information.txt'
    output_file = 'cleaned_information.txt'

    clean_data(input_file, output_file)








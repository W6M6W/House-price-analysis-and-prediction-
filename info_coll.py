import requests
import time
import random
from bs4 import BeautifulSoup


def fetch_property_info(max_pages=10):
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/115.0.0.0 Safari/537.36")
    }

    file_path = "total_information.txt"

    for page in range(1, max_pages + 1):
        # 根据页数生成对应的 URL：
        # 第一页为：https://hy.58.com/ershoufang/i11220yy4/
        # 后续页面为：https://hy.58.com/ershoufang/i11220p{page}yy4/
        if page == 1:
            url = "https://hy.58.com/ershoufang/i11220ra4yy4/"
        else:
            url = f"https://hy.58.com/ershoufang/i11220p{page}ra4yy4/"

        print(f"正在抓取第 {page} 页: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # 确保请求成功
        except Exception as e:
            print(f"请求第 {page} 页出错: {e}")
            continue  # 跳过当前页，继续下一页

        soup = BeautifulSoup(response.text, "html.parser")

        # 查找房源列表
        list_section = soup.find("section", class_="list-left")
        if not list_section:
            print(f"第 {page} 页未找到 list-left 区域")
            continue

        list_container = list_section.find("section", class_="list")
        if not list_container:
            print(f"第 {page} 页未找到 list 容器")
            continue

        properties = list_container.find_all("div", class_="property")
        if not properties:
            print(f"第 {page} 页未找到任何房源信息")
            continue

        with open(file_path, "a", encoding="utf-8") as file:
            for index, property_div in enumerate(properties, start=1):
                property_content = property_div.find("div", class_="property-content")
                if not property_content:
                    continue

                try:
                    # 获取房产名称
                    title_tag = property_content.find("h3", class_="property-content-title-name")
                    title = title_tag.text.strip() if title_tag else "未知"

                    # 获取房屋格局
                    attributes = property_content.find("p", class_="property-content-info-attribute")
                    rooms = attributes.find_all("span") if attributes else []
                    room_count = rooms[0].text.strip() if len(rooms) > 0 else "?"
                    hall_count = rooms[2].text.strip() if len(rooms) > 2 else "?"
                    bath_count = rooms[4].text.strip() if len(rooms) > 4 else "?"

                    # 获取面积
                    area_tag = attributes.find_next_sibling("p") if attributes else None
                    area = area_tag.text.strip().replace("㎡", "") if area_tag else "?"

                    # 获取朝向
                    direction_tag = area_tag.find_next_sibling("p") if area_tag else None
                    direction = direction_tag.text.strip() if direction_tag else "?"

                    # 获取楼层信息
                    floor_tag = direction_tag.find_next_sibling("p") if direction_tag else None
                    floor = floor_tag.text.strip() if floor_tag else "?"

                    # 获取建造年代
                    year_tag = floor_tag.find_next_sibling("p") if floor_tag else None
                    year = year_tag.text.strip().replace("年建造", "") if year_tag else "?"

                    # 获取地址
                    address_tag = property_content.find("p", class_="property-content-info-comm-address")
                    address = "".join([span.text.strip() for span in address_tag.find_all("span")]) if address_tag else "?"

                    # 获取总价
                    total_price_tag = property_content.find("span", class_="property-price-total-num")
                    total_price = total_price_tag.text.strip() if total_price_tag else "?"

                    # 获取单价
                    unit_price_tag = property_content.find("p", class_="property-price-average")
                    unit_price = unit_price_tag.text.strip().replace("元/㎡", "") if unit_price_tag else "?"

                    # 格式化输出
                    property_info = (f"编号：{index}，房产名称：“{title}”，格局：{room_count}室{hall_count}厅{bath_count}卫，"
                                     f"面积：{area}㎡，朝向：{direction}，楼层：{floor}，建造年代：{year}，"
                                     f"地址：{address}，总价：{total_price}万，单价：{unit_price}元/㎡")

                    print(property_info)

                    # 写入文件，每爬取一个房源就存储，防止数据丢失
                    file.write(property_info + "\n")

                except Exception as e:
                    print(f"解析第 {page} 页房源 {index} 时出错: {e}")

        # 访问过快可能会被封，添加随机等待时间（1-3秒）
        sleep_time = random.uniform(2, 5)
        print(f"等待 {sleep_time:.2f} 秒后继续...")
        time.sleep(sleep_time)


if __name__ == "__main__":
    fetch_property_info(max_pages=20)  # 这里设定最多爬取 10 页，可自行调整




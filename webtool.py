from main import evaluate_property_all
from main import final_work
from flask import Flask, request, jsonify
from flask_cors import CORS  # 新增导入

app = Flask(__name__)
CORS(app)

# 处理前端数据的函数
def data_deal(user_input):
    try:
        # 完整参数解析
        area = float(user_input.get("area"))
        direction = user_input.get("direction", "south")
        floor_level = user_input.get("floorLevel", "low")
        total_floors = int(user_input.get("totalFloors"))
        year_built = int(user_input.get("year"))
        address = user_input.get("address", "")

        # 朝向映射
        direction_mapping = {"south": '南', "north": '北', "east": '东', "west": '西'}
        # 楼层系数映射
        floor_level_mapping = {"high": '高层', "medium": '中层', "low": '低层'}

        # 构造数据数组
        clean_data = [
            1,  # 基础参数1
            422,  # 基础参数2，根据您期望的输出调整
            area,
            direction_mapping[direction],  # 朝向系数
            f"{floor_level_mapping[floor_level]}(共{total_floors}层)",  # 楼层系数
            year_built,  # 建筑年份
            address,
            660000,  # 根据您期望的输出设置的占位符
            4204  # 根据您期望的输出设置的占位符
        ]

        print("处理后的数据:", clean_data)
        return clean_data

    except Exception as e:
        print(f"数据处理出错: {e}")
        return None

# 接收前端请求并返回计算结果
@app.route('/evaluate', methods=['POST'])
def evaluate_price():
    try:
        # 获取用户提交的数据（JSON 格式）
        user_input = request.get_json()

        # 处理数据
        processed_data = data_deal(user_input)
        if processed_data is None:
            return jsonify({"error": "数据格式错误"}), 400

        # 计算房价和影响因素
        estimated_price = final_work(processed_data)
        return_data_nodeal = evaluate_property_all(
            processed_data)  # 返回 [orientation_metric, floor_total_metric, floor_category_metric, region_avg_price, whole_aver]
        whole_aver=return_data_nodeal[4]
        return_data=return_data_nodeal[:4]
        # 返回 JSON 结果，包含评估结果和影响因素数据
        return jsonify({"estimate": estimated_price, "factors": return_data,"whole_aver": whole_aver})
    except Exception as e:
        return jsonify({"error": f"服务器错误: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)








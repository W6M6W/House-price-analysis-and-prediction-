import random
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

# 调用各因素模块（假设这些模块已实现）
from a_to_price import a_p_evolution
from ad_to_price import ad_to_price_return, get_region_code
from year_to_price import year_to_price_return
from le_to_price import le_to_price_return
from d_to_price import drice_price_return
from ad_to_price import data_clean

###########################
# 数据加载与划分函数
###########################

def load_data_into_list(input_file):
    property_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = infile.readlines()
    for line in data:
        fields = line.strip().split()
        if len(fields) >= 9:
            property_data.append(fields)
    return property_data

def split_data(data, test_ratio=0.2, seed=42):
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    n_test = int(len(data_copy) * test_ratio)
    test_set = data_copy[:n_test]
    train_set = data_copy[n_test:]
    return train_set, test_set

###########################
# 统一房价预测模型（融合简单加权、多项式回归和决策树集成）
###########################

class UnifiedHousePriceModel:
    def __init__(self, training_data):
        """
        training_data: 二维数组，每行格式为 [编号, 款式, 面积, 朝向, 楼层, 建成年份, 地址, 总价, 单价]
        """
        self.training_data = training_data

        # 调用各因素模块获取全局参数
        self.area_params = a_p_evolution(training_data)
        self.address_adjustments = ad_to_price_return(training_data)
        self.year_params = year_to_price_return(training_data)
        self.floor_adjustments = le_to_price_return(training_data)
        self.orientation_adjustments = drice_price_return(training_data)

        # 各因素初始权重（简单加权预测用，保持不变）
        self.weights = {
            'area': 1.0,
            'address': 1.0,
            'year': 1.0,
            'floor': 1.0,
            'orientation': 1.0
        }

        # 模型组件
        self.poly_model = None
        self.poly_transform = None
        self.decision_tree_ensemble_model = None

        # 融合预测的初始权重：分别对应简单加权、多项式回归和决策树集成
        self.fusion_weights = np.array([0.45, 0.45, 0.10])

    def extract_features(self, house):
        """
        对单个房源数据提取各因素贡献特征，返回5维特征向量。
        house: [编号, 款式, 面积, 朝向, 楼层, 建成年份, 地址, 总价, 单价]
        """
        # 面积因素
        try:
            area = float(house[2])
        except:
            area = 0.0
        area_slope = self.area_params[0][0] if self.area_params and len(self.area_params[0]) > 0 else 0
        area_intercept = self.area_params[1] if self.area_params and len(self.area_params) > 1 else 0
        area_feature = area_slope * area + area_intercept

        # 地址因素
        region_code = get_region_code(house[6])
        if 0 <= region_code < len(self.address_adjustments):
            address_feature = self.address_adjustments[region_code]
        else:
            address_feature = 0

        # 建成年份因素
        try:
            year_val = int(house[5])
            if year_val > 0:
                year_feature = self.year_params[0] * math.log(year_val) + self.year_params[1]
            else:
                year_feature = 0
        except:
            year_feature = 0

        # 楼层因素：简单判断“高层”或“低层”
        floor_info = house[4]
        if "高层" in floor_info:
            floor_feature = self.floor_adjustments[0] if len(self.floor_adjustments) > 0 else 0
        elif "低层" in floor_info:
            floor_feature = self.floor_adjustments[1] if len(self.floor_adjustments) > 1 else 0
        else:
            floor_feature = 0

        # 朝向因素
        orientation = house[3]
        if orientation in ['南', '南北']:
            orientation_feature = self.orientation_adjustments[0] if len(self.orientation_adjustments) > 0 else 0
        else:
            orientation_feature = self.orientation_adjustments[1] if len(self.orientation_adjustments) > 1 else 0

        features = np.array([area_feature, address_feature, year_feature, floor_feature, orientation_feature])
        return features

    def predict_weighted(self, house):
        """
        使用各因素贡献的加权和（基于预设权重）来预测单价。
        """
        features = self.extract_features(house)
        prediction = (self.weights['area'] * features[0] +
                      self.weights['address'] * features[1] +
                      self.weights['year'] * features[2] +
                      self.weights['floor'] * features[3] +
                      self.weights['orientation'] * features[4])
        return prediction

    def build_polynomial_model(self, degree=2):
        """
        基于训练数据与提取的特征构建多项式回归模型，
        捕捉各因素间的非线性影响。
        """
        X, Y = [], []
        for house in self.training_data:
            features = self.extract_features(house)
            X.append(features)
            try:
                unit_price = float(house[-1])
            except:
                unit_price = 0
            Y.append(unit_price)
        X = np.array(X)
        Y = np.array(Y)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, Y)
        self.poly_model = model
        self.poly_transform = poly
        return model

    def predict_polynomial(self, house):
        """
        使用训练好的多项式回归模型对单个房源预测单价。
        """
        if self.poly_model is None or self.poly_transform is None:
            raise ValueError("请先调用 build_polynomial_model() 构建多项式模型。")
        features = self.extract_features(house).reshape(1, -1)
        X_poly = self.poly_transform.transform(features)
        return self.poly_model.predict(X_poly)[0]

    def build_decision_tree_ensemble_model(self, max_depth=None):
        """
        构造集成决策树模型：
        每个样本构造新的特征向量：[原始特征 (5维), 简单加权预测 (1维), 多项式预测 (1维)]。
        """
        X, y = [], []
        for house in self.training_data:
            original_features = self.extract_features(house)
            weighted_pred = self.predict_weighted(house)
            poly_pred = self.predict_polynomial(house)
            combined_features = np.concatenate([original_features, [weighted_pred, poly_pred]])
            X.append(combined_features)
            try:
                unit_price = float(house[-1])
            except:
                unit_price = 0
            y.append(unit_price)
        X = np.array(X)
        y = np.array(y)
        dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        dt_model.fit(X, y)
        self.decision_tree_ensemble_model = dt_model
        return dt_model

    def predict_decision_tree_ensemble(self, house):
        """
        使用集成决策树模型对单个房源进行预测。
        """
        if self.decision_tree_ensemble_model is None:
            raise ValueError("请先调用 build_decision_tree_ensemble_model() 构建集成决策树模型。")
        original_features = self.extract_features(house)
        weighted_pred = self.predict_weighted(house)
        poly_pred = self.predict_polynomial(house)
        combined_features = np.concatenate([original_features, [weighted_pred, poly_pred]])
        combined_features = combined_features.reshape(1, -1)
        return self.decision_tree_ensemble_model.predict(combined_features)[0]

    def predict_fusion(self, house):
        """
        融合预测方法：
        将简单加权、多项式回归和决策树集成的预测结果，
        按照融合权重加权求和，默认权重为 [0.45, 0.45, 0.10]。
        """
        S = self.predict_weighted(house)
        P = self.predict_polynomial(house)
        D = self.predict_decision_tree_ensemble(house)
        fusion_pred = np.dot(self.fusion_weights, np.array([S, P, D]))
        return fusion_pred

###########################
# 动态约束优化器（优化融合权重，要求权重在 [0,1] 内且归一化）
###########################

class DynamicConstraintOptimizer:
    def __init__(self, model):
        """
        model: UnifiedHousePriceModel 的实例
        """
        self.model = model
        self.weights = self.model.fusion_weights.copy()

    def optimize(self, train_data, epochs=50, lr=0.01):
        """
        利用梯度下降优化融合权重，同时施加动态约束。
        """
        for epoch in range(epochs):
            grad = np.zeros(3)
            loss = 0.0
            for house in train_data:
                S = self.model.predict_weighted(house)
                P = self.model.predict_polynomial(house)
                D = self.model.predict_decision_tree_ensemble(house)
                preds = np.array([S, P, D])
                try:
                    y = float(house[-1])
                except:
                    y = 0.0
                final_pred = np.dot(self.weights, preds)
                error = final_pred - y
                loss += error ** 2
                grad += 2 * error * preds
            loss /= len(train_data)
            grad /= len(train_data)
            self.weights -= lr * grad
            # 施加动态约束：权重限制在 [0,1] 内，并归一化
            self.weights = np.clip(self.weights, 0, 1)
            if np.sum(self.weights) != 0:
                self.weights /= np.sum(self.weights)
            else:
                self.weights = np.array([0.35, 0.35, 0.3])
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Weights = {self.weights}, Loss = {loss:.4f}")
        self.model.fusion_weights = self.weights.copy()
        return self.weights

###########################
# 遗传算法优化器（同时优化融合权重、多项式阶数和决策树最大深度）
###########################

class GeneticOptimizer:
    def __init__(self, model):
        self.model = model
        # 定义搜索范围
        self.weight_range = (0.0, 10.0)          # 融合权重候选值范围（后续归一化）
        self.poly_degree_range = (2, 5)          # 多项式模型阶数范围（整数）
        self.dt_depth_range = (3, 15)            # 决策树最大深度范围（整数）
        self.population_size = 10
        self.generations = 10
        self.mutation_rate = 0.1

    def evaluate_candidate(self, candidate, training_data):
        """
        candidate 为 5 维向量：[w1, w2, w3, poly_degree, dt_depth]
        """
        weights = candidate[:3]
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.array([0.45, 0.45, 0.10])
        poly_degree = int(round(candidate[3]))
        dt_depth = int(round(candidate[4]))
        # 构建模型
        self.model.build_polynomial_model(degree=poly_degree)
        self.model.build_decision_tree_ensemble_model(max_depth=dt_depth)
        # 暂时设置融合权重
        original_weights = self.model.fusion_weights.copy()
        self.model.fusion_weights = weights
        errors = []
        for house in training_data:
            pred = self.model.predict_fusion(house)
            try:
                y = float(house[-1])
            except:
                y = 0.0
            errors.append((pred - y) ** 2)
        mse = np.mean(errors)
        # 恢复原来的融合权重
        self.model.fusion_weights = original_weights
        return mse

    def mutate(self, candidate):
        new_candidate = candidate.copy()
        for i in range(len(candidate)):
            if random.random() < self.mutation_rate:
                if i < 3:
                    new_candidate[i] += random.uniform(-1, 1)
                    new_candidate[i] = max(self.weight_range[0], min(self.weight_range[1], new_candidate[i]))
                elif i == 3:
                    new_candidate[i] += random.uniform(-0.5, 0.5)
                    new_candidate[i] = max(self.poly_degree_range[0], min(self.poly_degree_range[1], new_candidate[i]))
                else:
                    new_candidate[i] += random.uniform(-1, 1)
                    new_candidate[i] = max(self.dt_depth_range[0], min(self.dt_depth_range[1], new_candidate[i]))
        return new_candidate

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def optimize(self, training_data):
        population = []
        for _ in range(self.population_size):
            candidate = np.array([
                random.uniform(self.weight_range[0], self.weight_range[1]),
                random.uniform(self.weight_range[0], self.weight_range[1]),
                random.uniform(self.weight_range[0], self.weight_range[1]),
                random.uniform(self.poly_degree_range[0], self.poly_degree_range[1]),
                random.uniform(self.dt_depth_range[0], self.dt_depth_range[1])
            ])
            population.append(candidate)
        best_candidate = None
        best_fitness = float('inf')
        for gen in range(self.generations):
            fitnesses = []
            for candidate in population:
                mse = self.evaluate_candidate(candidate, training_data)
                fitnesses.append(mse)
                if mse < best_fitness:
                    best_fitness = mse
                    best_candidate = candidate.copy()
            print(f"Generation {gen}, best MSE: {best_fitness:.4f}")
            sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
            survivors = sorted_population[:self.population_size // 2]
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population
        return best_candidate, best_fitness

###########################
# 堆叠集成（stacking）元模型
###########################

def build_stacking_model(training_data, unified_model):
    X_stack = []
    y_stack = []
    for house in training_data:
        S = unified_model.predict_weighted(house)
        P = unified_model.predict_polynomial(house)
        D = unified_model.predict_decision_tree_ensemble(house)
        X_stack.append([S, P, D])
        try:
            y_stack.append(float(house[-1]))
        except:
            y_stack.append(0.0)
    X_stack = np.array(X_stack)
    y_stack = np.array(y_stack)
    meta_model = LinearRegression()
    meta_model.fit(X_stack, y_stack)
    return meta_model

def stacking_predict(house, unified_model, meta_model):
    S = unified_model.predict_weighted(house)
    P = unified_model.predict_polynomial(house)
    D = unified_model.predict_decision_tree_ensemble(house)
    X = np.array([S, P, D]).reshape(1, -1)
    return meta_model.predict(X)[0]

###########################
# 主程序
###########################
def final_work(input):
    # 数据加载与划分
    input_file = 'cleaned_information.txt'
    property_data = load_data_into_list(input_file)
    train_set, test_set = split_data(property_data, test_ratio=0.2)
    print(f"训练集数量: {len(train_set)}, 测试集数量: {len(test_set)}")

    # 初始化统一模型
    unified_model = UnifiedHousePriceModel(train_set)
    # 构建多项式模型与决策树集成模型，初始超参数：多项式阶数 2，决策树最大深度 5
    unified_model.build_polynomial_model(degree=2)
    unified_model.build_decision_tree_ensemble_model(max_depth=5)

    # 输出单个样本的各基模型预测
    sample_house = input
    print("样本房源:", sample_house)
    simple_prediction = unified_model.predict_weighted(sample_house)
    poly_prediction = unified_model.predict_polynomial(sample_house)
    dt_prediction = unified_model.predict_decision_tree_ensemble(sample_house)
    fusion_prediction = unified_model.predict_fusion(sample_house)
    print("简单加权预测单价：", simple_prediction)
    print("多项式回归预测单价：", poly_prediction)
    print("决策树集成预测单价：", dt_prediction)
    print("融合预测单价（初始权重）：", fusion_prediction)

    # 第一阶段：利用动态约束优化器优化融合权重
    optimizer = DynamicConstraintOptimizer(unified_model)
    optimized_weights = optimizer.optimize(train_set, epochs=100, lr=0.01)
    print("动态约束优化后的融合权重：", optimized_weights)
    fusion_prediction_optimized = unified_model.predict_fusion(sample_house)
    print("融合预测单价（动态优化后）：", fusion_prediction_optimized)

    # 第二阶段：利用遗传算法优化融合权重和超参数（多项式阶数、决策树最大深度）
    genetic_optimizer = GeneticOptimizer(unified_model)
    best_candidate, best_fitness = genetic_optimizer.optimize(train_set)
    print("遗传算法优化最佳候选解：", best_candidate)
    print("最佳候选解适应度 (MSE)：", best_fitness)
    # 更新模型参数
    best_weights = best_candidate[:3]
    if np.sum(best_weights) > 0:
        best_weights = best_weights / np.sum(best_weights)
    else:
        best_weights = np.array([0.45, 0.45, 0.10])
    best_poly_degree = int(round(best_candidate[3]))
    best_dt_depth = int(round(best_candidate[4]))
    unified_model.fusion_weights = best_weights
    unified_model.build_polynomial_model(degree=best_poly_degree)
    unified_model.build_decision_tree_ensemble_model(max_depth=best_dt_depth)
    fusion_prediction_genetic = unified_model.predict_fusion(sample_house)
    print("融合预测单价（遗传算法优化后）：", fusion_prediction_genetic)

    # 第三阶段：构建堆叠集成元模型
    meta_model = build_stacking_model(train_set, unified_model)
    stacking_final_pred = stacking_predict(sample_house, unified_model, meta_model)
    stacking_final_pred=int(stacking_final_pred)
    print("堆叠集成最终预测单价：", stacking_final_pred)
    return stacking_final_pred


def evaluate_property_all(input_row ):
    input_file = 'cleaned_information.txt'
    property_data = load_data_into_list(input_file)
    """
    输入：
      - input_row: 单条房源记录，例如：
          [1, 422, 120.0, '南', '高层(共22层)', 2022, '衡阳市华新大道花园小区701', 660000, 4204]
      - sample_data: 样本集，使用 load_data_into_list 加载后的数据列表

    输出一个列表，包含四个指标：
      1. 朝向指标：若房源朝向为 '南' 或 '南北'，则取 drice_price_return(sample_data) 的第一项；否则取第二项
      2. 楼层总数指标：从 input_row[4] 中提取总层数，如果大于 8 层，取 le_to_price_return(sample_data) 的第一项；否则取第二项
      3. 楼层类别指标：若楼层描述中包含 '高层'，取 le_to_price_return(sample_data) 的第三项；若包含 '中层'，取第四项；否则返回 0
      4. 地址指标：调用 get_region_code 确定地址所属区域，然后利用 data_clean(sample_data) 计算该区域的平均房价
    """
    # 1. 朝向指标
    # drice_price_return 返回 [percent_increase_sn, percent_decrease_other]
    drice_result = drice_price_return(property_data)
    if input_row[3] in ['南', '南北']:
        orientation_metric = drice_result[0]
    else:
        orientation_metric = drice_result[1]

    # 2. 楼层总数指标
    floor_info = input_row[4]  # 如 '高层(共22层)'
    try:
        total_floors = int(floor_info.split('共')[1].split('层')[0])
    except (IndexError, ValueError):
        total_floors = 0
    le_result = le_to_price_return(property_data)  # 返回 8 个百分比指标
    floor_total_metric = le_result[0] if total_floors > 8 else le_result[1]

    # 3. 楼层类别指标
    if '高层' in floor_info and  total_floors > 8:
        floor_category_metric = le_result[2]
    elif '中层' in floor_info and  total_floors > 8:
        floor_category_metric = le_result[3]
    elif '低层' in floor_info and  total_floors > 8:
        floor_category_metric = le_result[4]
    elif '高层' in floor_info and  total_floors <= 8:
        floor_category_metric = le_result[5]
    elif '中层' in floor_info and  total_floors <= 8:
        floor_category_metric = le_result[6]
    elif '低层' in floor_info and  total_floors <= 8:
        floor_category_metric = le_result[7]
    else:
        floor_category_metric = 0

    # 4. 地址指标
    # 利用 get_region_code 确定地址所属区域
    region_code = get_region_code(input_row[6])
    # 通过 data_clean 对样本数据进行清洗，得到 [ad_id, region_code, price] 格式的数据
    cleaned_data = data_clean(property_data)
    # 计算目标区域内所有房源的平均房价
    region_prices = [price for _, r_code, price in cleaned_data if r_code == region_code]
    region_avg_price = sum(region_prices) / len(region_prices) if region_prices else 0
    whole_aver_list=[int(group[-1]) for group in property_data]
    whole_aver=np.mean(whole_aver_list)
    orientation_metric=int(orientation_metric*100)
    floor_total_metric=int(floor_total_metric*100)
    floor_category_metric=int(floor_category_metric*100)
    region_avg_price=int(region_avg_price)
    whole_aver=int(whole_aver)

    return [orientation_metric, floor_total_metric, floor_category_metric, region_avg_price,whole_aver]




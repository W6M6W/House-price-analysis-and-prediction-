### **House Price Analysis and Prediction — 房价分析与预测项目**  

---

## **🏠 Introduction — 项目介绍**  

This is an intelligent tool designed specifically for real estate valuation, combining web crawling technology, data processing capabilities, influencing factor analysis, and advanced machine learning and data analysis methods. It can automatically collect rich real estate reference resources from your preferred target websites, covering key data such as transaction information, market trends, and geographical advantages of various types of real estate.  
这是一款专为房产估值设计的智能工具，结合了网络爬虫技术、数据处理能力、影响因素分析以及先进的机器学习和数据分析手段，能够自动从您喜好的目标网站上抓取丰富的房产参照资源，这些资源涵盖了各类房产的交易信息、市场趋势、地理位置优势等关键数据。  

Through in-depth data processing and precise analysis of influencing factors, this software provides insights into subtle changes in the real estate market, capturing every detail that affects housing prices. Based on this, we employ powerful machine learning algorithms to generate a highly accurate valuation tool for real estate.  
通过深度的数据处理和精细的影响因素分析，软件能够洞察房产市场的微妙变化，把握影响房价的每一个细节。在此基础上，我们利用强大的机器学习算法，生成了一套精准的房产估值工具。

Users only need to input specific details about the property they want to sell—such as house area, layout, renovation condition, surrounding facilities, etc. The software quickly analyzes this data and provides a scientifically reasonable estimate of the house price. More importantly, the software offers a detailed explanation of the logic and reasoning behind the valuation, helping users understand the basis of the price estimation clearly.  

用户只需输入想要出售房屋的具体信息，如房屋面积、户型、装修状况、周边设施等，软件便可快速进行综合分析，并提供科学合理的房价预测。更重要的是，软件还会详细解释估价背后的逻辑和依据，让用户能够清楚理解房价预测的依据，做到心中有数。  

---

## **🔍 Project Steps — 项目步骤**  

1️⃣ **Data Collection & Cleaning 数据收集与清理**  
- Scrape real estate data from housing market websites and process the raw data to obtain a structured dataset.  从房屋销售网站爬取房产数据，并处理原始数据，以获得结构化数据集。 
- Relevant scripts相关脚本: `info_coll.py`, `info_manage.py`  

2️⃣ **Factor Analysis 因素分析**  
- Analyze the impact of each factor on the housing price per square meter separately.  分析各个因素对每平方米房价的影响，包括房屋面积、地址、装修情况、户型以及建造年份等。 
- Relevant scripts相关脚本: `a_to_price.py` (area), `ad_to_price.py` (address), `d_to_price.py` (decoration), `le_to_price.py` (layout), `year_to_price.py` (year built)  

3️⃣ **Model Training & Optimization 模型训练与优化**  
- Integrate factor analysis results using simple weighting, polynomial regression, and decision tree ensembles.  结合因素分析结果，运用简单加权、多项式回归、决策树集成等方法进行整合。
- Use gradient descent to optimize fusion weights while applying dynamic constraints.  通过梯度下降优化融合权重，并应用动态约束。
- Implement genetic algorithms for hyperparameter tuning and ensemble model stacking.  利用遗传算法进行超参数调优，并构建集成学习的元模型。
- Relevant script相关脚本: `main.py`  

4️⃣ **Web Interface & Deployment Web 界面与部署**  
- Develop a web interface that allows users to input property details and get real-time price predictions. 开发交互式网页，用户可直接输入房产信息，实时获取房价预测结果。   
- Support bilingual switching between English and Chinese.  支持中英文双语切换。 
- Relevant files: `webtool.py`, `webpage.html`, `script.js`, `style.css`  
  

---

## **🎯 Final Effect — 最终效果**  
  
- Real-time housing price prediction based on user-input property details.  根据用户输入的房产信息进行实时房价预测。
- Transparent analysis with detailed reasoning behind the valuation results.  提供透明的分析结果，并详细解释估值逻辑。
- Interactive web application with an intuitive and user-friendly interface. 交互式网页应用，界面直观，易于操作。 
- Multi-language support (Chinese & English) for a wider audience.  支持多语言（中英文）切换，适用于更广泛的用户群体。 

-   inputweb：![屏幕截图 2025-02-20 193348](https://github.com/user-attachments/assets/f626dd8c-27cb-4692-b6d6-d098d6b8843f)

-   waitingpage:![屏幕截图 2025-02-20 193434](https://github.com/user-attachments/assets/319d7f0c-bc42-4d5c-926e-2aa7493852d4)

-   outweb:![image](https://github.com/user-attachments/assets/74bae7b6-8683-4270-b369-7eaf46559a0b)

-  

---

## **⚠️ Precautions — 注意事项**  
 
- Ensure that all required dependencies are installed before running the project.  在运行项目之前，请确保已安装所有必要的依赖项。
- Key dependencies include Python libraries such as Flask, NumPy, Pandas, Scikit-learn, and BeautifulSoup.  主要依赖项包括 Python 库，如 Flask、NumPy、Pandas、Scikit-learn 和 BeautifulSoup。 
- If running locally, install dependencies using 如果在本地运行，请使用以下命令安装依赖项:  
  ```bash
  pip install -r requirements.txt
  ```  
- Make sure to configure the web scraping module properly according to the target website’s structure.  确保根据目标网站的结构正确配置网络爬虫模块，以便成功获取数据。
 

---

## **📜 License — 许可证**  
This project is released under the **MIT License**, meaning you are free to use, modify, and distribute it with proper attribution.  

本项目基于 **MIT 许可证** 开源，您可以自由使用、修改和分发，但请保留原作者信息。  

---

## **💡 Contribution — 贡献指南**  

### **English Version:**  
We welcome contributions! If you’d like to improve the project, please follow these steps:  
1. **Fork** the repository.  
2. **Create** a new branch (`git checkout -b feature-branch`).  
3. **Commit** your changes (`git commit -m "Add new feature"`).  
4. **Push** the branch (`git push origin feature-branch`).  
5. **Submit** a pull request.  


---

This structured, bilingual README will help your project stand out on GitHub! 🚀

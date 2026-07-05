# User Behavior Analysis (用户行为分析项目)

## 📝 项目简介
本项目旨在通过 Python 对用户行为数据进行深度挖掘与分析。主要包含数据预处理、用户价值分层（RFM模型）、转化路径分析以及客户流失预测等核心功能，为业务决策提供数据支持。

## 🚀 主要功能模块

本项目包含三个核心脚本，分别对应不同的分析场景：

1. **data_clean&RFM.py**
   - **功能**：数据清洗与用户价值建模。
   - **内容**：
     - 原始数据的缺失值处理、异常值过滤及格式标准化。
     - 构建 RFM 模型（Recency, Frequency, Monetary），对用户进行分层（如重要保持客户、一般发展客户等）。

2. **Conversion_Funnel_Analysis.py** (转化漏斗分析)
   - **功能**：全链路转化效率评估。
   - **内容**：
     - 定义关键业务节点（如：浏览 -> 加购 -> 下单 -> 支付）。
     - 计算各环节转化率，识别流失严重的瓶颈环节。

3. **Customer_Churn_Warning.py** (客户流失预警)
   - **功能**：潜在流失用户识别。
   - **内容**：
     - 基于用户历史行为特征构建流失预测模型。
     - 输出高流失风险用户名单，辅助运营团队进行精准挽留。

## 🛠️ 环境依赖
运行上述脚本需要安装以下 Python 库：
- pandas
- numpy
- matplotlib / seaborn (用于可视化)
- scikit-learn (用于流失预警模型)

可以通过以下命令安装依赖：
pip install pandas numpy matplotlib seaborn scikit-learn

## 📂 文件结构
User_Behavior_Analysis/
├── data_clean&RFM.py          # 数据清洗与RFM分析
├── Conversion_Funnel_An...    # 转化漏斗分析
└── Customer_Churn_Warn...     # 客户流失预警

## 📊 如何使用
1. 确保你的工作目录下有对应的数据集（如 `.csv` 或 `.xlsx` 文件）。
2. 修改脚本中的文件读取路径（例如 `pd.read_csv('your_data.csv')`）。
3. 依次运行脚本查看分析结果或生成的图表。

---
*Author: maojiadong*

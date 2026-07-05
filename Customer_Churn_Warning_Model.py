"""
项目模块：Cox比例风险生存分析 - 用户流失预警建模
数据集：data/user_rfm_segmented.csv（RFM分层后的用户行为指标）
建模说明：
    1. 建模时剔除recency特征，仅使用用户长期行为衍生特征训练模型
    2. T=用户最后活跃间隔，E标记是否存在后续交互（删失标记）
输出：user_churn_prediction.csv，包含用户流失风险分、风险等级
"""
import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# ====================== 全局路径配置 ======================
# 输入：RFM分层结果（上级目录data文件夹）
INPUT_RFM = os.path.join('..', 'data', 'user_rfm_segmented.csv')
# 输出：带流失风险评分与风险等级的全量用户表
OUTPUT_CHURN = "user_churn_prediction.csv"

print(" 步骤3：流失预警模型构建启动...")
# 1. 读取第一步生成的用户画像文件
try:
    df = pd.read_csv(INPUT_RFM)
    print(f" 数据加载成功，包含 {len(df)} 个用户。")
except FileNotFoundError:
    print(f" 错误：找不到数据文件 {INPUT_RFM}，请先运行RFM分层脚本生成数据！")
    exit()

# 2. 特征工程（优化：填充空值，防止对数变换报错）
# 衍生业务特征
df['conversion_rate'] = df['freq'] / (df['pv_count'] + 1)  # 下单转化率
df['active_ratio'] = df['freq'] / (df['recency'] + 1)      # 日均下单活跃密度

# 对计数型特征做log(1+x)平滑，缓解长尾偏态分布
features_to_transform = ['freq', 'pv_count', 'cart_count', 'fav_count', 'conversion_rate']
for col in features_to_transform:
    df[col] = df[col].fillna(0)  # 填充缺失值
    df[f'log_{col}'] = np.log1p(df[col])

# 3. 构造生存分析标签（T=生存时长，E=事件发生标记）
max_recency = df['recency'].max()
df['T'] = df['recency']
# E=1：用户存在后续交互（发生观测事件）；E=0：达到最大间隔，视为删失样本
df['E'] = (df['recency'] < max_recency).astype(int)

# 4. 建模特征集【核心修复：移除recency，消除数据泄露】
# 仅保留行为衍生特征，不使用与目标T同源的recency
model_columns = [
    'T', 'E',
    'log_freq', 'log_pv_count', 'log_cart_count', 'log_fav_count',
    'log_conversion_rate'
]

# 校验字段完整性
missing_cols = [c for c in model_columns if c not in df.columns]
if missing_cols:
    print(f" 警告：数据集缺失建模所需字段：{missing_cols}，请检查RFM输出文件！")
    exit()
df_model = df[model_columns].copy()

# 5. 训练带L2正则的Cox比例风险模型，防止过拟合
cph = CoxPHFitter(penalizer=0.1)
try:
    cph.fit(df_model, duration_col='T', event_col='E')

    # 输出模型统计摘要
    print("\n 模型回归摘要 (coef系数 & 风险比exp(coef) & 显著性p值):")
    cph.print_summary(columns=["coef", "exp(coef)", "p"])

    # 打印各特征流失风险影响排序
    print("\n===== 特征对流失风险影响排序（exp(coef)>1会提升流失风险） =====")
    coef_result = cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
    print(coef_result.round(4))

    # 计算一致性指数C-index
    c_index = concordance_index(
        df_model['T'],
        -cph.predict_partial_hazard(df_model),
        df_model['E']
    )
    print(f"\n 模型 C-index: {c_index:.4f}")
    print("解读：0.5=随机猜测，0.6~0.7具备基础区分力，0.7以上模型效果良好")

    # 6. 计算用户流失风险分 + 划分高/中/低流失风险等级
    df['risk_score'] = cph.predict_partial_hazard(df_model)
    # 三等分风险分层
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[0, df['risk_score'].quantile(0.33), df['risk_score'].quantile(0.66), df['risk_score'].max()],
        labels=['低流失风险', '中流失风险', '高流失风险']
    )
    print("\n===== 用户流失风险分层分布 =====")
    print(df['risk_level'].value_counts())

    # 7. 保存完整结果表
    df.to_csv(OUTPUT_CHURN, index=False, encoding="utf-8-sig")
    print(f"\n 全量用户流失预测结果已保存至 {OUTPUT_CHURN}")

except Exception as e:
    print(f" 模型训练失败，异常信息：{e}")
    print("排查提示：1. 检查特征是否存在无穷值 2. 检查特征完全多重共线性 3. 确认数据无空值")
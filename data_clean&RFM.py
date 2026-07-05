import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
FILE_PATH = os.path.join('..', 'data', 'UserBehavior.csv')
OUTPUT_FILE = 'user_rfm_segmented.csv'

print(" 步骤1：数据清洗与RFM分层启动...")

try:
    # 1. 读取数据
    df = pd.read_csv(FILE_PATH, header=None,
                     names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])

    print(f"原始数据量: {len(df)} 行")

    # 将时间戳转换为 datetime
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')

    # 设定合理的时间窗口 (淘宝 UserBehavior 数据集的标准时间是 2017-11-25 到 2017-12-03)
    start_date = pd.to_datetime('2017-11-01')
    end_date = pd.to_datetime('2017-12-31')

    # 过滤掉不在该时间段内的异常数据
    before_count = len(df)
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    after_count = len(df)

    print(f" 时间清洗完成：移除了 {before_count - after_count} 条异常时间数据")
    print(f" 有效数据时间跨度: {df['time'].min()} 至 {df['time'].max()}")

    # 3. 核心指标聚合 (计算每个用户的 R, F, 以及行为计数)
    # 我们以数据的最大时间作为“当前时间”来计算 Recency
    max_time = df['time'].max()

    print(" 正在计算用户统计指标...")

    user_stats = df.groupby('user_id').agg(
        # R (Recency): 最近一次行为距离最大时间的天数
        recency=('time', lambda x: (max_time - x.max()).days),

        # F (Frequency): 购买次数 (behavior_type == 'buy')
        freq=('behavior_type', lambda x: (x == 'buy').sum()),

        # 辅助特征：浏览次数
        pv_count=('behavior_type', lambda x: (x == 'pv').sum()),

        # 辅助特征：加购次数
        cart_count=('behavior_type', lambda x: (x == 'cart').sum()),

        # 辅助特征：收藏次数
        fav_count=('behavior_type', lambda x: (x == 'fav').sum())
    ).reset_index()

    # 4. RF分层逻辑(原数据无M，固仅采用RF这个简化模型)
    # 定义阈值 (使用中位数作为分割线，比固定值更稳健)
    r_thresh = user_stats['recency'].median()
    f_thresh = user_stats['freq'].median()


    # 简单的四象限打标
    def label_user(row):
        if row['recency'] <= r_thresh and row['freq'] > f_thresh:
            return '重要价值用户 (高活高频)'
        elif row['recency'] <= r_thresh and row['freq'] <= f_thresh:
            return '重要发展用户 (高活低频)'
        elif row['recency'] > r_thresh and row['freq'] > f_thresh:
            return '重要保持用户 (低活高频)'
        else:
            return '一般挽留用户 (低活低频)'


    user_stats['user_label'] = user_stats.apply(label_user, axis=1)

    # 5. 保存结果
    user_stats.to_csv(OUTPUT_FILE, index=False)
    print(f"\n处理完成！结果已保存至: {OUTPUT_FILE}")
    print("-" * 30)
    print("用户分层概览:")
    print(user_stats['user_label'].value_counts())

except FileNotFoundError:
    print(f"错误：找不到文件 {FILE_PATH}，请检查路径。")
except Exception as e:
    print(f"发生未知错误: {e}")
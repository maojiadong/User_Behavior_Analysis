import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
print("步骤2：漏斗分析与转化归因启动...")

# 读取数据
DATA_PATH = os.path.join('..', 'data', 'UserBehavior.csv')
df = pd.read_csv(DATA_PATH, header=None, names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])

# ========== 统计各行为【独立用户数】==========
user_behavior_cnt = df.groupby('behavior_type')['user_id'].nunique()

# 主转化漏斗标准递进链路：浏览 -> 加购 -> 购买
main_steps = ['pv', 'cart', 'buy']
main_step_users = [user_behavior_cnt.get(s, 0) for s in main_steps]

# 收藏为并行行为，单独提取
fav_user_num = user_behavior_cnt.get('fav', 0)

# ========== 分别计算两种转化率 ==========
# 1. 相邻环节环比转化率（当前/上一步）
step_convert = []
for idx in range(len(main_step_users)):
    if idx == 0:
        step_convert.append(1.0)
    else:
        prev = main_step_users[idx-1]
        curr = main_step_users[idx]
        step_convert.append(curr / prev if prev > 0 else 0)

# 2. 全局转化率（各环节用户 / 总浏览用户）
pv_total = main_step_users[0]
global_convert = [num / pv_total if pv_total > 0 else 0 for num in main_step_users]

# ========== 可视化：主转化漏斗柱状图+环比转化率折线 ==========
fig, (ax1, ax_fav) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：主转化漏斗 pv-cart-buy
sns.barplot(x=main_steps, y=main_step_users, palette="Blues_d", ax=ax1)
ax1.set_ylabel('独立用户数', fontsize=12)
ax1.set_title('淘宝用户核心转化漏斗（浏览-加购-购买）', fontsize=14)
# 柱子标注用户数量
for i, v in enumerate(main_step_users):
    ax1.text(i, v + max(main_step_users)*0.01, f'{int(v)}', ha='center', fontweight='bold')

# 双Y轴绘制环比转化率
ax2 = ax1.twinx()
ax2.plot(main_steps, step_convert, color='red', marker='o', linewidth=2, label='环节环比转化率')
ax2.set_ylabel('环比转化率', fontsize=12, color='red')
ax2.set_ylim(0, 1.1)
# 标注环比百分比
for i, v in enumerate(step_convert):
    ax2.text(i, v + 0.02, f'{v:.2%}', ha='center', color='red', fontweight='bold')

# 右图：并行行为收藏用户对比
behavior_labels = ['浏览pv', '收藏fav']
behavior_nums = [pv_total, fav_user_num]
sns.barplot(x=behavior_labels, y=behavior_nums, palette="Greens", ax=ax_fav)
ax_fav.set_title('收藏行为独立用户规模', fontsize=14)
ax_fav.set_ylabel('独立用户数')
for i, v in enumerate(behavior_nums):
    ax_fav.text(i, v + max(behavior_nums)*0.01, f'{int(v)}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ========== 输出关键指标与业务结论 ==========
print("\n===== 漏斗核心转化指标 =====")
print(f"总浏览独立用户：{main_step_users[0]}")
print(f"加购独立用户：{main_step_users[1]}")
print(f"购买独立用户：{main_step_users[2]}")
print(f"收藏独立用户：{fav_user_num}")
print("-" * 40)
print(f"浏览 → 加购 环比转化率：{step_convert[1]:.2%}")
print(f"加购 → 购买 环比转化率：{step_convert[2]:.2%}")
print("-" * 40)
print(f"浏览全局转加购：{global_convert[1]:.2%}")
print(f"浏览全局转购买：{global_convert[2]:.2%}")
print("-" * 40)

# 业务诊断建议
if step_convert[1] < 0.1:
    print("【流失预警】浏览→加购转化率低于10%，流失严重")
    print("优化建议：优化商品主图/详情页、新增优惠券、首页个性化推荐")
elif step_convert[2] < 0.2:
    print("【流失预警】加购→购买转化率偏低，加购用户大量未下单")
    print("优化建议：加购降价提醒、限时折扣、简化结算流程、减少运费")
else:
    print("转化链路整体健康，可重点提升收藏用户的复购与下单转化")
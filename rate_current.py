import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 从log.txt中提取的实际数据
# 接受新模型并评估基线的迭代
accepted_iterations = [1, 4, 10, 18, 19, 27, 29, 30]
# 评估结果 (从log.txt可以看到所有接受的新模型对基线都是完胜)
wins_vs_baseline = [20] * len(accepted_iterations)
loses_vs_baseline = [0] * len(accepted_iterations)
draws_vs_baseline = [0] * len(accepted_iterations)

# 计算胜率和不输率
win_rates = [win/20 for win in wins_vs_baseline]
no_lose_rates = [(win + draw)/20 for win, draw in zip(wins_vs_baseline, draws_vs_baseline)]

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(accepted_iterations, win_rates, marker='o', linestyle='-', linewidth=2, markersize=8, label='胜率')
plt.plot(accepted_iterations, no_lose_rates, marker='s', linestyle='-', linewidth=2, markersize=8, color='green', label='不输率(胜率+平局率)')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)  # 添加0.5基准线

# 添加图表标题和标签
plt.title('AlphaZero对基线模型的胜率与不输率变化', fontsize=16)
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('比率', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0.4, 1.05)  # 调整Y轴范围以更好地显示完美胜率
plt.legend(fontsize=12)

# 优化显示效果
plt.xticks(accepted_iterations)
plt.tight_layout()

# 为每个数据点添加标签
for i, (x, y) in enumerate(zip(accepted_iterations, win_rates)):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

# 保存图表
plt.savefig('alphazero_performance_current.png', dpi=300)
plt.show()
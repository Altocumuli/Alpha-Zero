import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 迭代次数
iterations = list(range(8, 31))

# 从日志中提取的胜率数据
win_counts = [12, 10, 13, 15, 10, 12, 17, 16, 15, 13, 9, 10, 14, 19, 17, 14, 11, 12, 12, 16, 13, 17, 17]
lose_counts = [8, 9, 7, 5, 8, 8, 3, 3, 5, 7, 9, 10, 6, 1, 2, 5, 8, 7, 7, 4, 6, 3, 2]
draw_counts = [0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]

# 计算胜率和不输率
win_rates = [win/20 for win in win_counts]
no_lose_rates = [(win + draw)/20 for win, draw in zip(win_counts, draw_counts)]

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(iterations, win_rates, marker='o', linestyle='-', linewidth=2, markersize=8, label='胜率')
plt.plot(iterations, no_lose_rates, marker='s', linestyle='-', linewidth=2, markersize=8, color='green', label='不输率')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)  # 添加0.5基准线

# 添加图表标题和标签
plt.title('AlphaZero对基线模型的胜率与不输率变化', fontsize=16)
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('比率', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0.4, 1.0)
plt.legend(fontsize=12)

# 优化显示效果
plt.xticks(iterations)
plt.tight_layout()

# 保存图表
plt.savefig('alphazero_performance.png', dpi=300)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 17:11:45 2025

@author: qinxiaotian
"""

import matplotlib.pyplot as plt
import re

# 读取日志文件
log_file = "train_data.txt"  # 替换为你的日志文件路径
with open(log_file, 'r') as f:
    lines = f.readlines()

# 解析日志数据
epochs = []
iterations = []
costs = []
global_steps = []

# 正则表达式匹配模式
pattern = r"Epoch (\d+),iteration (\d+):cost=([\d.]+)"

for line in lines:
    match = re.match(pattern, line.strip())
    if match:
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        cost = float(match.group(3))
        
        # 计算全局步数（从0开始）
        global_step = epoch * 6000 + (iteration - 1)  # 假设iteration从1开始计数
        
        epochs.append(epoch)
        iterations.append(iteration)
        costs.append(cost)
        global_steps.append(global_step)

# 创建图表
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.plot(global_steps, costs, linewidth=1.0)

# 添加标签和标题
plt.xlabel("Training Step (Epoch × Iteration)", fontsize=12)
plt.ylabel("Cost", fontsize=12)
plt.title("Training Loss Curve", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加epoch分隔线
for epoch in range(1, 5):
    plt.axvline(x=epoch*6000, color='r', linestyle='--', alpha=0.5)
    plt.text(epoch*6000+3000, max(costs)*0.9, f'Epoch {epoch}', 
             horizontalalignment='center', fontsize=10)

# 优化坐标轴范围
plt.xlim(0, max(global_steps))
plt.ylim(min(costs)*0.95, max(costs)*1.05)

# 保存并显示图表
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()
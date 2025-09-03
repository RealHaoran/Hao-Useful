import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
n_samples = 500              # 总样本数
pT_values = [0.9, 0.5, 0.1]  # 目标概率的不同取值
mean = [0, 0]                # 正态分布的均值
cov = [[25, 0], [0, 25]]     # 正态分布的协方差矩阵 (diag(25,25))
a, b = [-10, -10], [10, 10]  # 均匀分布的范围 [a, b]

# 创建画布，包含两行三列的三维子图
fig = plt.figure(figsize=(20, 14))
axes = []
for i in range(2):    # 两行：混合模型和线性组合模型
    row = []
    for j in range(3):  # 三列：pT=0.9, 0.5, 0.1
        ax = fig.add_subplot(2, 3, i*3 + j + 1, projection='3d')
        row.append(ax)
    axes.append(row)
axes = np.array(axes)
plt.subplots_adjust(wspace=0.2, hspace=0.2)

# 混合模型：绘制到第一行的子图
for col_idx, pT in enumerate(pT_values):
    ax = axes[0, col_idx]  # 第一行的子图
    pC = 1 - pT
    
    # 生成目标点和杂波点
    target_mask = np.random.rand(n_samples) < pT
    num_target = np.sum(target_mask)
    num_clutter = n_samples - num_target
    
    target_samples = np.random.multivariate_normal(mean, cov, num_target)
    clutter_samples = np.random.uniform(low=a, high=b, size=(num_clutter, 2))
    
    # 绘制目标和杂波点，z轴区分类别：目标z=1，杂波z=0
    ax.scatter(
        target_samples[:, 0], target_samples[:, 1], np.ones(num_target),  # z=1
        c='red', label='Target', alpha=0.7, edgecolors='white', s=40, depthshade=False
    )
    ax.scatter(
        clutter_samples[:, 0], clutter_samples[:, 1], np.zeros(num_clutter),  # z=0
        c='blue', label='Clutter', alpha=0.7, edgecolors='white', s=40, depthshade=False
    )
    
    # 设置子图标题和坐标范围
    ax.set_title(f'Mixture Model: $P_{{T}}$ = {pT}', fontsize=12)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-0.5, 1.5)  # 限制z轴范围以突出层次
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.view_init(elev=25, azim=-45)  # 设置视角（仰角25度，方位角-45度）
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

# 线性组合模型：绘制到第二行的子图
for col_idx, pT in enumerate(pT_values):
    ax = axes[1, col_idx]  # 第二行的子图
    pC = 1 - pT
    
    # 生成正态和均匀分布的样本
    x = np.random.multivariate_normal(mean, cov, n_samples)
    y = np.random.uniform(low=a, high=b, size=(n_samples, 2))
    z = pT * x + pC * y
    
    # 绘制线性组合点，z=0.5表示模型类型
    ax.scatter(
        z[:, 0], z[:, 1], np.full(n_samples, 0.5),  # z=0.5
        c='green', alpha=0.7, edgecolors='black', s=40, depthshade=False
    )
    
    # 设置子图标题和坐标范围
    ax.set_title(f'Linear Combination: $P_{{T}}$ = {pT}', fontsize=12)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-0.5, 1.5)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.view_init(elev=25, azim=-45)
    ax.grid(True, linestyle='--', alpha=0.5)

# 添加全局标题并显示图像
fig.suptitle('3D Visualization of Radar Measurements (Mixture vs. Linear Combination)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
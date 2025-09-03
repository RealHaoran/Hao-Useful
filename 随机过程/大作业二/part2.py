# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 14,           # 基础字体
    'axes.titlesize': 14,      # 子图标题字体
    'axes.labelsize': 14,      # 坐标轴标签字体
    'xtick.labelsize': 14,     # x轴刻度字体
    'ytick.labelsize': 14,     # y轴刻度字体
    'legend.fontsize': 14,     # 图例字体
    'figure.titlesize': 18     # 总标题字体
})

# 设置实验参数
num_experiments = 10000
sample_sizes = [1, 2, 5, 10, 30, 50, 100]

# 定义新的分布类型及其生成函数
distributions = {
    'Poisson(λ=3)': lambda n: np.random.poisson(3, n),
    'Chi-squared(df=2)': lambda n: np.random.chisquare(2, n),
    'Beta(a=2,b=5)': lambda n: np.random.beta(2, 5, n)
}

# 创建图形窗口
plt.figure(figsize=(18, 10))

# 对每种分布和样本容量绘图
for dist_idx, (dist_name, dist_func) in enumerate(distributions.items()):
    for size_idx, n in enumerate(sample_sizes):
        sample_means = np.zeros(num_experiments)
        for i in range(num_experiments):
            sample_data = dist_func(n)
            sample_means[i] = np.mean(sample_data)
        
        # 子图排列
        plt.subplot(len(distributions), len(sample_sizes), dist_idx * len(sample_sizes) + size_idx + 1)
        
        # 绘制直方图与核密度估计
        sns.histplot(sample_means, kde=True, stat="density", bins=30, color='skyblue')
        
        # 绘制理论正态分布曲线
        x_values = np.linspace(np.min(sample_means), np.max(sample_means), 100)
        normal_pdf = (1 / (np.std(sample_means) * np.sqrt(2 * np.pi))) * \
                     np.exp(-(x_values - np.mean(sample_means))**2 / (2 * np.std(sample_means)**2))
        plt.plot(x_values, normal_pdf, 'r--', lw=2)
        
        # 设置标题和标签，字体调大
        plt.title(f'{dist_name}\nn={n}', fontsize=15)
        plt.xlabel('Sample Mean', fontsize=14)
        plt.ylabel('Density', fontsize=14)

# 总标题与布局
plt.suptitle('CLT Verification: Poisson, Chi-squared, Beta Distributions', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

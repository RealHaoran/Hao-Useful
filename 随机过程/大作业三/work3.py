import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 设置matplotlib显示负号正常
plt.rcParams['axes.unicode_minus'] = False  

# 1. 加载数据
data = pd.read_excel('作业三/GPdata.xlsx', header=None, names=['x', 'y'])
X = data['x'].values.reshape(-1, 1)
y = data['y'].values

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 高斯过程回归模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.04, n_restarts_optimizer=10, random_state=42)
gp.fit(X_train, y_train)

# 4. 生成预测点并预测
X_plot = np.linspace(-10, 10, 1000).reshape(-1, 1)
y_pred, sigma = gp.predict(X_plot, return_std=True)
y_pred_test = gp.predict(X_test)
gp_mse = mean_squared_error(y_test, y_pred_test)

# 5. 多项式回归比较
degrees = [1, 3, 5, 7]
poly_mses = []

plt.figure(figsize=(15, 10))
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_poly = model.predict(X_plot)
    y_poly_test = model.predict(X_test)
    mse_poly = mean_squared_error(y_test, y_poly_test)
    poly_mses.append(mse_poly)
    
    plt.subplot(2, 2, i+1)
    plt.scatter(X_train, y_train, c='k', s=10, label='Training Data')
    plt.plot(X_plot, y_poly, 'b-', label=f'Degree {degree} Polynomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Degree {degree} Polynomial Fit (MSE={mse_poly:.4f})')
    plt.legend()

plt.tight_layout()
plt.show()

# 6. 绘制高斯过程回归结果
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, c='k', s=10, label='Training Data')
plt.plot(X_plot, y_pred, 'r-', label='GP Mean Prediction')
plt.fill_between(X_plot.ravel(), 
                y_pred - 1.96*sigma, 
                y_pred + 1.96*sigma, 
                alpha=0.2, color='red', label='95% Confidence Interval')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Gaussian Process Regression (MSE={gp_mse:.4f})')
plt.legend()
plt.show()

# 7. 打印结果比较
print("\n=== Model Comparison ===")
print(f"Gaussian Process Regression Test MSE: {gp_mse:.4f}")
for degree, mse in zip(degrees, poly_mses):
    print(f"Degree {degree} Polynomial Fit Test MSE: {mse:.4f}")

# 8. 打印高斯过程优化参数
print("\n=== Gaussian Process Optimized Parameters ===")
print("Optimized Kernel:", gp.kernel_)
print("Log Marginal Likelihood:", gp.log_marginal_likelihood_value_)
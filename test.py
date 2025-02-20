import numpy as np
import matplotlib.pyplot as plt


# 定义函数
def f(x):
    return 236.6817 * x ** 4 - 2.5534 * x ** 3 + 0.0119 * x ** 2 - 2010.53957309


# 生成x值
x = np.linspace(-2, 2, 1000)  # 在区间[-2, 2]内生成1000个点

# 计算y值
y = f(x)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$y=236.6817x^4 - 2.5534x^3 + 0.0119x^2 - 2010.53957309$')
plt.title('Function Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
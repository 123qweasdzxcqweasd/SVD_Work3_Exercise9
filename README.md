# SVD 图像压缩

基于奇异值分解（Singular Value Decomposition）的图像低秩逼近与压缩演示。

## 原理

对图像矩阵 $A \in \mathbb{R}^{m \times n}$ 做奇异值分解：

$$A = U \Sigma V^T$$

只保留前 $k$ 个最大奇异值进行重构：

$$A \approx U_k \Sigma_k V_k^T$$

**存储量对比**

| 项目 | 存储量 |
| --- | --- |
| 原始图像 | $m \times n$ |
| 压缩后 | $k(m + n + 1)$ |

当 $k \ll \min(m, n)$ 时可获得显著压缩，且由于大奇异值承载了图像的主要能量，视觉损失很小。

## 环境依赖

- Python 3.8+
- NumPy
- Matplotlib
- Pillow

安装：

```bash
pip install numpy matplotlib pillow
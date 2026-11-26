import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']   # Windows 用 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方框的问题

# ---------- 1. 读取图片并转为灰度 ----------
img = Image.open("zyy_image.jpg")   # 换成你自己的图片路径
img = img.convert("L")               # 转灰度
A = np.array(img, dtype=np.float64)  # 转为 float 矩阵

m, n = A.shape
print(f"图像尺寸: {m} x {n}")

# ---------- 2. 奇异值分解 ----------
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"奇异值个数: {len(S)}, 最大奇异值: {S[0]:.2f}")

# ---------- 3. 低秩逼近函数 ----------
def low_rank_approx(k):
    """取前 k 个奇异值重构图像"""
    k = min(k, len(S))
    Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return Ak

# ---------- 4. 计算压缩率 ----------
def compression_ratio(k):
    original = m * n
    compressed = k * (m + n + 1)   # U前k列 + S前k个 + Vt前k行
    return compressed / original

# ---------- 5. 画图 ----------
ks = [50, 10]
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

axes[0].imshow(A, cmap="gray")
axes[0].set_title("(a) 原图")
axes[0].axis("off")

for i, k in enumerate(ks):
    Ak = low_rank_approx(k)
    ratio = compression_ratio(k)
    axes[i + 1].imshow(Ak, cmap="gray")
    axes[i + 1].set_title(f"提取 {k} 个奇异值\n存储占比 {ratio:.2%}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig("svd_compress.png", dpi=150)
plt.show()

# ---------- 6. 奇异值衰减曲线（可选，写报告很有用）----------
plt.figure(figsize=(8, 5))
plt.plot(S[:100], "b.-")
plt.title("前 100 个奇异值衰减曲线")
plt.xlabel("奇异值序号")
plt.ylabel("奇异值大小")
plt.grid(True)
plt.savefig("singular_values.png", dpi=150)
plt.show()
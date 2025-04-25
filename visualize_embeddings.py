#!/usr/bin/env python3
"""
File name: visualize_embeddings.py
Purpose: Load the trained model, extract the 2D embedding of the dataset, visualize it in a scatterplot.
Supported when visualizing:
    1) Autoencoder model (encoder extracts 2D features)
    2) Contrast learning models (contrastive_color / contrastive_shape)
"""

import os
import glob
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ---------- 1) Define Encoder structure consistent with training scripts -----------
class AE_Encoder(nn.Module):
    """与 autoencoder_shapes.py 中的Encoder对应。"""
    def __init__(self):
        super(AE_Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.net(x)

class Contrastive_Encoder(nn.Module):
    """与 contrastive_shapes.py 中的Encoder对应。"""
    def __init__(self):
        super(Contrastive_Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        z = self.net(x)
        # 对比学习里我们一般做归一化
        z = F.normalize(z, dim=1)
        return z

def load_autoencoder_encoder(model_path, device):
    """加载 autoencoder_shapes.py 训练的 encoder 参数"""
    # 先构建同样的结构
    encoder = AE_Encoder().to(device)
    # 读取权重
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["encoder"] if "encoder" in ckpt else ckpt
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder

def load_contrastive_encoder(model_path, device):
    """加载 contrastive_shapes.py 训练的 encoder 参数"""
    encoder = Contrastive_Encoder().to(device)
    state_dict = torch.load(model_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder

# ---------- 2) Read the images in the dataset and resolve their shape/color labels ----------

def load_images_and_labels(data_dir, device, sample_size=3000):
    """
    加载图像并解析标签，但只随机抽样 sample_size 张，避免一次性 OOM。
    返回：
      - images_tensor: Tensor [M,3,64,64]
      - shape_labels: list of length M
      - color_labels: list of length M
    """
    # 1) 列出所有文件并随机抽样
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if len(all_files) > sample_size:
        all_files = random.sample(all_files, sample_size)

    # 2) 解析标签、读图并 transform
    shape_labels = []
    color_labels = []
    images = []
    transform_vis = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for f in all_files:
        # 解析 shape/color
        basename = os.path.basename(f)
        parts = basename.split("_")
        shape_labels.append(parts[0])
        color_labels.append(int(parts[1]))

        # 读取并预处理图像
        img = Image.open(f).convert("RGB")
        img_t = transform_vis(img).unsqueeze(0)  # [1,3,64,64]
        images.append(img_t)

    # 3) 合并并迁移到 device
    images_tensor = torch.cat(images, dim=0).to(device)  # [M,3,64,64]
    return images_tensor, shape_labels, color_labels

# ---------- 3) Generate Scatterplot ----------
def visualize_scatter(embeddings, shape_labels, color_labels, title="z-space", outpath="embedding_vis.png"):
    """
    embeddings: (N,2) numpy
    shape_labels: list of str  [circle|square|triangle]
    color_labels: list of int  (0~7)
    """
    plt.figure(figsize=(8,8))
    # 给每种 shape 指定一个 marker
    markers = {'circle': 'o', 'square': 's', 'triangle': '^'}

    cmap = cm.get_cmap('tab10',8)  # 颜色映射，假设 color_idx 在 [0,9]
    unique_shapes = sorted(list(set(shape_labels)))

    # ---- 在这之前，先随机抽样 ----
    n_samples = 500
    sel = np.random.choice(len(embeddings), n_samples, replace=False)
    emb = embeddings[sel]
    sh  = [shape_labels[i]  for i in sel]
    co  = [color_labels[i]  for i in sel]

    # ---- 然后用 emb, sh, co 画图 ----
    for shape in set(sh):
        idxs = [i for i,s in enumerate(sh) if s==shape]
        # cols = [cmap(co[i]) for i in idxs]
        cols = [cmap(co[i] / 7.0) for i in idxs]
        plt.scatter(emb[idxs,0],
                    emb[idxs,1],
                    c=cols,
                    marker=markers[shape],
                    s=120,       # 大一点
                    alpha=0.9,   # 也可更不透明
                    label=shape)

    plt.title(title)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    # plt.legend()
    # plt.grid(True)
    plt.savefig(outpath, dpi=300)
    plt.show()
    print(f"Visual Scatterplot saved: {outpath}")

# ---------- 4) main function ----------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入图像 & 标签
    images, shape_labels, color_labels = load_images_and_labels(args.data_dir, device)

    # 根据模式加载对应模型
    if args.mode == "autoencoder":
        # 从 autoencoder.pth 加载 encoder
        encoder = load_autoencoder_encoder(args.model_path, device)
        # encoder 输出是 [B,2,4,4] 还需要 flatten
        # 这里做个简化：将输出 reshape到 (B, 2*4*4) 再做个 normalize
        with torch.no_grad():
            z = encoder(images)  # [N,2,4,4]
            B, C, H, W = z.shape
            z = z.view(B, -1)    # (B,2*4*4)= (B,32)
            # 不一定一定要做normalize，但做了可视化对比更直观
            z = F.normalize(z, dim=1)
            # 这里为了和对比学习一样得到2D，可自行选择:
            #   1) 直接用 [B,32] 做 PCA到 2D
            #   2) 在 AE 结构里就把 embedding 输出到 2D (如 stride=4)
            # 若想要最简洁地2D，就在 AE 里最后直接输出 (B,2) =>
            #  参考 autoencoder_shapes.py 也可以那样设计
            # 下面示例假设 AE 最后一层是 (B,2,4,4)，那可以用global pool或flatten再减少到2D
            # 这里先简单演示 flatten 后再手动 PCA:
            z_np = z.cpu().numpy()  # (N,32)
        # 如果需要真的严格 2D，则做一下PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_np)  # => (N,2)
        embeddings = z_2d
        outname = "embedding_vis_autoencoder.png"

    elif args.mode == "contrastive_color" or args.mode == "contrastive_shape":
        # 加载 contrastive encoder
        encoder = load_contrastive_encoder(args.model_path, device)
        with torch.no_grad():
            z = encoder(images)  # (N,2)
        embeddings = z.cpu().numpy()
        if args.mode == "contrastive_color":
            outname = "embedding_vis_color.png"
        else:
            outname = "embedding_vis_shape.png"

    else:
        raise ValueError("mode must be [autoencoder|contrastive_color|contrastive_shape]")

    # 绘制散点图
    visualize_scatter(embeddings, shape_labels, color_labels,
                      title=f"Embeddings: {args.mode}",
                      outpath=outname)


    # ==============================================================
#  Add-on: generates a “query + k nearest neighbors” visualization (e.g., right side of Fig. 30.14)
# ==============================================================

def show_knn_grid(embeddings,            # (N,2) numpy
                  file_list,             # 与 embeddings 对应的路径 list[str]
                  k=6,                   # 每个 query 显示几个邻居
                  num_queries=6,         # 显示多少行 query
                  save_name="knn_grid.png"):
    """
    Generate a (num_queries × (k+1)) grid plot:
        The leftmost of each row is the query, the next k are the nearest neighbors (by cosine similarity)
    """
    assert len(file_list) == embeddings.shape[0]
    # 1. 随机选 query 下标
    q_idx = np.random.choice(len(file_list), num_queries, replace=False)
    # 2. 归一化后做余弦距离
    Z = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = Z @ Z.T  # (N,N)
    # 3. 为每个 query 找 k 个最大相似度（排除自身）
    rows = []
    for qi in q_idx:
        sims = sim[qi]
        nn_idx = sims.argsort()[::-1][1:k+1]   # 跳过自身
        rows.append([qi] + nn_idx.tolist())

    # 4. 画图
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.figure(figsize=((k+1)*2, num_queries*2))
    for r, idx_row in enumerate(rows):
        for c, img_idx in enumerate(idx_row):
            ax = plt.subplot(num_queries, k+1, r*(k+1)+c+1)
            ax.axis("off")
            img = Image.open(file_list[img_idx]).convert("RGB")
            plt.imshow(img)
            if c == 0:
                ax.set_ylabel("query", rotation=0, labelpad=30,
                              fontsize=12, va='center')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Saved Nearest Neighbor Visualization：{save_name}")


# --------------------------------------------------------------
#  Add a --knn trigger to the CLI (to maintain backwards compatibility)
# --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="shape_dataset")
    parser.add_argument("--mode",
                        choices=["autoencoder",
                                 "contrastive_color",
                                 "contrastive_shape"],
                        default="contrastive_shape")
    parser.add_argument("--model_path", default="contrastive_shape.pth")
    parser.add_argument("--knn", action="store_true",
                        help="若给出，则画 query‑NN 网格")
    args = parser.parse_args()

    # ===== 先调用前面的 main() 生成 embeddings =====
    images, shape_labels, color_labels = load_images_and_labels(args.data_dir,
                                                                torch.device("cpu"),
                                                                sample_size=64000)
    encoder = (load_contrastive_encoder if "contrastive" in args.mode
               else load_autoencoder_encoder)(args.model_path, "cpu")
    with torch.no_grad():
        Z = encoder(images).cpu().numpy()

    # scatter
    visualize_scatter(Z, shape_labels, color_labels,
                      title=f"Embeddings: {args.mode}",
                      outpath=f"embedding_vis_{args.mode}.png")

    # 最近邻可视化
    if args.knn:
        file_list = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
        show_knn_grid(Z, file_list,
                      k=6, num_queries=6,
                      save_name=f"knn_{args.mode}.png")


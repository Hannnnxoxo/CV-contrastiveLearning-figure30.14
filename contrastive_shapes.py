#!/usr/bin/env python3
"""
file name: contrastive_shapes.py
Purpose: Learning a 2D representation using contrast learning (InfoNCE) for colored geometric data (similar to book 30.10.2)"""

import os
import glob
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------
# 1) Dataset Classes (Constructing Positive Sample Pairs)
# ---------------------------
class ContrastiveShapeDataset(Dataset):
    def __init__(self, image_dir, transform1, transform2):
        """
        After reading each image, two views are obtained by transform1 and transform2 -> (view1, view2)
        Used as a positive sample pair for InfoNCE (different enhancements of the same image)
        """
        self.image_dir = image_dir
        self.image_files = glob.glob(os.path.join(image_dir, "*.png"))
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        v1 = self.transform1(img)
        v2 = self.transform2(img)
        return v1, v2

# ---------------------------
# 2) Encoder structure: output 2-dimensional and normalized
# ---------------------------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 简单CNN做下采样，最后全局池化 -> 2D输出
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # -> 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # ->128x8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# ->256x4x4
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), # ->256x1x1
            nn.Flatten(),                # ->256
            nn.Linear(512, 2)            # ->2
        )
    def forward(self, x):
        z = self.net(x)
        # 归一化，使点分布在单位球面 (alignment+uniformity)
        z = F.normalize(z, dim=1)
        return z

# ---------------------------
# 3) InfoNCE loss function
# ---------------------------
def info_nce_loss(z1, z2, temperature=0.5):
    """
    计算InfoNCE损失:
      - z1, z2: (batch, embedding_dim)
      - 正样本对为 z1[i] 与 z2[i]
      - 其他 (z1[i], z2[j]) j!=i 均视为负样本
    """
    

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)      # (2N, dim)
    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)      # 排除 diagonal

    pos_idx = torch.arange(batch_size, device=z.device)
    pos = torch.cat([pos_idx + batch_size, pos_idx], dim=0)
    # 交叉熵：每行的正例是 pos[i]
    loss = F.cross_entropy(sim, pos)
    return loss






def uniformity(z, t=2.0):
    pdist2 = torch.pdist(z, p=2).pow(2)  # (B*(B-1)/2,)
    return torch.log(torch.exp(-t * pdist2).mean())


# ---------------------------
# 4) Main process of training
# ---------------------------
def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据view选择两种增强策略
    if args.view == "color":
        # 颜色敏感 => 仅裁剪/翻转，不做颜色扰动
        transform1 = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        transform2 = transform1  # 同样的变换策略
        suffix = "color"
    elif args.view == "shape":
       
        # ---- 书中设定：轻裁剪 + 颜色(HSV)抖动，**不要**翻转/灰度 ----
        def make_shape_transform():
            return transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                # transforms.RandomRotation(degrees=20),             

                transforms.ColorJitter(
                    brightness=0.8,
                    contrast=0.8,
                    saturation=1.0,
                   hue=0.5),
                transforms.RandomGrayscale(p=0.7),

                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        transform1 = make_shape_transform()
        transform2 = make_shape_transform()  # 两个独立对象 => 每次随机参数不同

        
        suffix = "shape"
    else:
        raise ValueError("view parameters should be 'color' or 'shape'")

    
    dataset = ContrastiveShapeDataset(args.data_dir, transform1, transform2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Start Comparative Learning (view={args.view}), steps={args.steps}, batch={args.batch_size}")
    step = 0
    while step < args.steps:
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)

            loss_i = info_nce_loss(z1, z2, temperature=args.temperature)
            z_all = torch.cat([z1, z2], dim=0)
            loss_u = uniformity(z_all, t=2.0)
            loss = loss_i + 0.1 * loss_u   # λ=0.1

            # loss=loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 500 == 0:
                print(f"Step {step}/{args.steps}  InfoNCE={loss_i:.4f}  Unif={loss_u:.4f}  Total={loss:.4f}")
            if step >= args.steps:
                break
    print("Contrast learning complete, save model...")
    torch.save(model.state_dict(), f"contrastive_{suffix}.pth")


    print("Comparative learning training ends!")

    # save model
    model_path = f"contrastive_{suffix}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved comparative learning model weights: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="shape_dataset", help="数据集路径")
    parser.add_argument("--view", type=str, default="color", help="选择 'color' 或 'shape'")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--steps", type=int, default=20000, help="对比学习总迭代次数")
    parser.add_argument("--temperature", type=float, default=0.5, help="InfoNCE温度超参数")
    args = parser.parse_args()
    main(args)

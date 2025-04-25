#!/usr/bin/env python3
"""
File name: generate_shape_dataset.py
Purpose: Generate a synthetic dataset containing colored geometric shapes for subsequent experiments
Description:
    - Image size: 64x64
    - Each image contains one geometric figure (circle, square or triangle)
    - The color of each figure is randomly selected from a list of preset colors (8 in total)
    - Each figure is randomly positioned, randomly sized, and randomly rotated.
    - Output nomenclature: {shape}_{colorIdx}_img_{i:05d}.png
"""
import math
import os
import random
from PIL import Image, ImageDraw
import argparse

# Define image size
IMG_SIZE = 64

# Define all possible shape types
SHAPE_TYPES = ['circle', 'square', 'triangle']

# Define a list of selectable colors (RGB format), 8 in total
COLOR_LIST = [
    (255, 0, 0),    # 红
    (0, 255, 0),    # 绿
    (0, 0, 255),    # 蓝
    (255, 255, 0),  # 黄
    (255, 0, 255),  # 洋红
    (0, 255, 255),  # 青
    (128, 0, 128),  # 紫罗兰
    (255, 165, 0)   # 橙
]

def generate_shape_image(shape_type, color, size=IMG_SIZE):
    """
Generates a 64x64 RGB image based on the specified shape type and color.    """
    # 创建一张黑色背景的新图像
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)


    # shape_radius 代表半径或半边长，随机
    shape_radius = random.randint(size // 8, size // 4)
    # shape_radius = size // 5
    margin = math.ceil(shape_radius * math.sqrt(2)) 
    center_x = random.randint(margin, size - margin)
    center_y = random.randint(margin, size - margin)
    
    
    
    # 绘制指定形状
    if shape_type == 'circle':
        bbox = [
            center_x - shape_radius, center_y - shape_radius,
            center_x + shape_radius, center_y + shape_radius
        ]
        draw.ellipse(bbox, fill=color)

    elif shape_type == 'square':
        bbox = [
            center_x - shape_radius, center_y - shape_radius,
            center_x + shape_radius, center_y + shape_radius
        ]
        draw.rectangle(bbox, fill=color)

    elif shape_type == 'triangle':
        point1 = (center_x, center_y - shape_radius)
        point2 = (center_x - shape_radius, center_y + shape_radius)
        point3 = (center_x + shape_radius, center_y + shape_radius)
        draw.polygon([point1, point2, point3], fill=color)
    else:
        raise ValueError("unknow shape：{}".format(shape_type))
    
    # 随机旋转图像
    angle = random.uniform(0, 360)
    try:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    except TypeError:
        # 如果 Pillow 版本过低，则省略 fillcolor 参数
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    return img

def generate_dataset(num_samples, output_dir):
    """
    Generate a specified number of images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_samples):
        shape_type = random.choice(SHAPE_TYPES)
        color_idx = random.randint(0, len(COLOR_LIST) - 1)
        color = COLOR_LIST[color_idx]
        
        img = generate_shape_image(shape_type, color, IMG_SIZE)
        
        filename = f"{shape_type}_{color_idx}_img_{i:05d}.png"
        img_path = os.path.join(output_dir, filename)
        img.save(img_path)
        
        if (i+1) % 1000 == 0:
            print(f"Generated {i+1} / {num_samples} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=64000,
                        help="要生成的图像数量，默认64000以更接近书中设置")
    parser.add_argument("--output_dir", type=str, default="shape_dataset",
                        help="图像输出保存文件夹")
    args = parser.parse_args()
    
    print(f"Start making dataset， {args.num_samples} images in all...")
    generate_dataset(args.num_samples, args.output_dir)
    print(f"The dataset is generated and saved in the {args.output_dir} folder.")

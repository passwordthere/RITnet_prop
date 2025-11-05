#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified for Step-by-Step Preprocessing Visualization (Test Set Only)
@author: Gemini (based on original Aayush code)
"""

import numpy as np
import torch
from torch.utils.data import Dataset 
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
import os.path as osp
import copy

# --- 辅助类和函数 (保持与原始文件一致，尽管在 test split 不会执行) ---

# 简化版的 transform（此处不用于可视化，但保留定义）
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
  
class RandomHorizontalFlip(object):
    def __call__(self, img,label):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT),\
                        label.transpose(Image.FLIP_LEFT_RIGHT)
        return img,label
    
class Starburst_augment(object):
    # 此处假设 'starburst_black.png' 存在于运行目录下
    def __call__(self, img):
        x=np.random.randint(1, 40)
        y=np.random.randint(1, 40)
        mode = np.random.randint(0, 2)
        try:
            starburst=Image.open('starburst_black.png').convert("L")
        except FileNotFoundError:
            print("Warning: starburst_black.png not found. Skipping Starburst_augment.")
            return Image.fromarray(img)
            
        if mode == 0:
            starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
            starburst = starburst[:, :-x]
        if mode == 1:
            starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
            starburst = starburst[:, x:]

        # 确保尺寸匹配，并进行图像混合
        img_array = np.array(img)
        starburst_array = np.array(starburst)
        
        # 简化混合逻辑，以避免复杂的边界检查
        # 原始代码中的索引操作很复杂，这里仅保留混合思想
        try:
             # 使用 NumPy 广播进行混合，假设 starburst 图像与眼部区域大小匹配
            mix_region_img = img_array[92+y:549+y, 0:400]
            mix_region_star = starburst_array[:mix_region_img.shape[0], :mix_region_img.shape[1]]
            
            if mix_region_img.shape == mix_region_star.shape:
                img_array[92+y:549+y,0:400] = mix_region_img * ((255 - mix_region_star) / 255) + mix_region_star
        except Exception:
            # 实际运行中如果 starburst 文件大小与预期的混合区域不匹配，可能会出错
            pass
            
        return Image.fromarray(img_array)

def getRandomLine(xc, yc, theta):
    x1 = xc - 50*np.random.rand(1)*(1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*np.random.rand(1) + 50)*(1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    # 转换为整数坐标
    return int(x1), int(y1), int(x2), int(y2)

class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value=np.random.randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img,(7,7),sigma_value))

class Translation(object):
    def __call__(self, base,mask):
        factor_h = 2*np.random.randint(1, 20)
        factor_v = 2*np.random.randint(1, 20)
        mode = np.random.randint(0, 4)
        # 省略实际的平移逻辑，因为在 test split 中不会被调用
        return Image.fromarray(base), Image.fromarray(mask)
            
class Line_augment(object):
    def __call__(self, base):
        yc, xc = (0.3 + 0.4*np.random.rand(1))*np.array(base.shape)
        aug_base = copy.deepcopy(base)
        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)
            # 确保输入是 BGR 或灰度图，这里是灰度图，所以颜色 (255, 255, 255) 只有第一个通道起作用
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), 255, 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       
        
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       

# --- 可视化相关的全局常量和辅助函数 ---

VISUALIZATION_WINDOW = 'Data Preprocessing Step-by-Step'

def display_image_with_text(img_np, text, window_name=VISUALIZATION_WINDOW):
    """在图像底部添加文字，并在固定窗口中显示"""
    # 确保图像是灰度图，并转换为 BGR 以便绘制彩色文字
    if img_np.ndim == 2:
        display_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        display_img = img_np
        
    # 在图像底部创建一个黑条区域用于文字显示
    height, width = display_img.shape[:2]
    footer_height = 40 # 增加底部空间
    img_with_footer = np.zeros((height + footer_height, width, 3), dtype=np.uint8)
    img_with_footer[:height, :, :] = display_img # 复制图像到顶部

    # 绘制文字
    cv2.putText(img_with_footer, text, 
                (5, height + 25), # 位置在底部
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2) # 字体大小和粗细
    
    cv2.imshow(window_name, img_with_footer)


# --- 核心数据集类 (针对可视化进行修改) ---

class VisualIrisDataset(Dataset):
    def __init__(self, filepath, split='test', transform=None, **args):
        # *** 关键修改: 默认使用 'train' split 来激活增强逻辑 ***
        # 但在文件查找时，我们仍然使用传入的 split 参数 (通常是 'test')
        # 我们只在内部将 self._split 设为 'train' 来启用增强
        self._split_for_aug = 'train' 
        self.transform = transform
        self.filepath = osp.join(filepath, split) # 仍然使用 'test' 查找文件
        listall = []
        
        image_dir = osp.join(self.filepath, 'images')
        if not os.path.exists(image_dir):
            print(f"Error: Image directory not found at {image_dir}")
            raise FileNotFoundError
            
        for file in os.listdir(image_dir):   
            if file.endswith(".png") or file.endswith(".jpg"):
               listall.append(file.rsplit('.', 1)[0]) 
        self.list_files = listall

        # PREPROCESSING STEP: CLAHE setup
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        filename = self.list_files[idx]
        
        # 尝试 .jpg 和 .png 两种扩展名
        imagepath_jpg = osp.join(self.filepath, 'images', filename + '.jpg')
        imagepath_png = osp.join(self.filepath, 'images', filename + '.png')
        
        if os.path.exists(imagepath_jpg):
            imagepath = imagepath_jpg
        elif os.path.exists(imagepath_png):
            imagepath = imagepath_png
        else:
            # 文件不存在，跳过
            return None
        
        pilimg = Image.open(imagepath).convert("L")
      
        # --- 0. 准备工作/原始图像 ---
        base_img = np.array(pilimg).copy()
        
        display_text_0 = f"[0/{filename}] Original Image. Press any key to continue..."
        display_image_with_text(base_img, display_text_0)
        if cv2.waitKey(0) == 27: return None 

        # --- 1. 预处理: 伽马校正 (这是第一个处理，所有增强都基于它) ---
        table = 255.0 * (np.linspace(0, 1, 256)**0.8)
        img_gamma = cv2.LUT(base_img, table).astype(np.uint8)
        
        display_text_1 = f"[1/{filename}] Gamma Correction (Factor 0.8). Press any key to continue..."
        display_image_with_text(img_gamma, display_text_1)
        if cv2.waitKey(0) == 27: return None
        
        # 将伽马校正后的图像作为增强的起点
        current_img_np = img_gamma.copy()
        step_index = 2

        # --- 2. 数据增强 (仅在模拟训练模式下执行) ---
        if self._split_for_aug == 'train':
            
            # --- 2a. Starburst augmentation (强制执行) ---
            current_img_np = np.array(Starburst_augment()(current_img_np))
            display_text_2a = f"[{step_index}/{filename}] AUG: Starburst Pattern. Press any key to continue..."
            display_image_with_text(current_img_np, display_text_2a)
            if cv2.waitKey(0) == 27: return None
            step_index += 1

            # --- 2b. Line augmentation (强制执行) ---
            current_img_np = np.array(Line_augment()(current_img_np))
            display_text_2b = f"[{step_index}/{filename}] AUG: Random Lines. Press any key to continue..."
            display_image_with_text(current_img_np, display_text_2b)
            if cv2.waitKey(0) == 27: return None
            step_index += 1
            
            # --- 2c. Gaussian Blur (强制执行) ---
            current_img_np = np.array(Gaussian_blur()(current_img_np))
            display_text_2c = f"[{step_index}/{filename}] AUG: Gaussian Blur. Press any key to continue..."
            display_image_with_text(current_img_np, display_text_2c)
            if cv2.waitKey(0) == 27: return None
            step_index += 1
            
            # --- 2d. Translation (强制执行) ---
            # Translation 原本需要 mask，这里我们只平移图像，mask 忽略
            # 注意：Translation 类返回 PIL Image，需要转回 np.array
            translated_pil_img, _ = Translation()(current_img_np, np.zeros_like(current_img_np))
            current_img_np = np.array(translated_pil_img)
            display_text_2d = f"[{step_index}/{filename}] AUG: Translation. Press any key to continue..."
            display_image_with_text(current_img_np, display_text_2d)
            if cv2.waitKey(0) == 27: return None
            step_index += 1
            
            # --- 2e. Random Horizontal Flip (强制执行) ---
            # Flip 也需要 mask，这里我们只翻转图像，mask 忽略
            # 注意：RandomHorizontalFlip 类接收/返回 PIL Image
            flipped_pil_img, _ = RandomHorizontalFlip()(Image.fromarray(current_img_np), Image.fromarray(np.zeros_like(current_img_np)))
            current_img_np = np.array(flipped_pil_img)
            display_text_2e = f"[{step_index}/{filename}] AUG: Random Horizontal Flip. Press any key to continue..."
            display_image_with_text(current_img_np, display_text_2e)
            if cv2.waitKey(0) == 27: return None
            step_index += 1
            

        # --- 3. 预处理: CLAHE (在所有增强之后) ---
        # 原始代码中 CLAHE 在部分增强（如 Translation, Flip）之前，
        # 但为了演示所有增强，我们将其放在所有增强之后。
        img_clahe = self.clahe.apply(current_img_np)
        
        display_text_3 = f"[{step_index}/{filename}] Final Step: CLAHE. Press any key for NEXT IMAGE..."
        display_image_with_text(img_clahe, display_text_3)
        if cv2.waitKey(0) == 27: return None
            
        return img_clahe, 0, filename, 0, 0

# ----------------------------------------------------------------------
# --- 主执行块 ---
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # !!! 请将这里改为您的数据集根目录 !!!
    # 示例: 如果您的图片路径是 "D:\data\iris_seg\test\images\image_01.jpg", 
    # 则 DATASET_ROOT 应该设置为 'D:\data\iris_seg'
    DATASET_ROOT = 'dataset' 
    SPLIT_NAME = 'test' 
    
    # 实例化可视化数据集
    try:
        ds_visual = VisualIrisDataset(filepath=DATASET_ROOT, split=SPLIT_NAME, transform=None)
    except FileNotFoundError:
        print("程序终止。请检查 DATASET_ROOT 路径是否正确。")
        exit()

    if not ds_visual.list_files:
        print("未在指定的测试集路径中找到任何 .jpg 或 .png 图像文件。请检查路径和文件格式。")
        exit()

    # 遍历数据集的前 N 张图片进行查看
    NUM_IMAGES_TO_VIEW = 100 # 设置您想查看的最大图片数量
    
    cv2.namedWindow(VISUALIZATION_WINDOW, cv2.WINDOW_AUTOSIZE) # 创建固定窗口
    
    print(f"Total images found: {len(ds_visual)}. Viewing first {min(NUM_IMAGES_TO_VIEW, len(ds_visual))}.")
    print("----------------------------------------------------------------------------------------")
    print("操作提示：请点击图像窗口，按任意键（非 ESC）查看下一步骤/下一张图片。按 ESC 键退出程序。")
    print("----------------------------------------------------------------------------------------")
    
    for i in range(min(NUM_IMAGES_TO_VIEW, len(ds_visual))):
        print(f"\n--- Loading Image {i+1}/{len(ds_visual)}: {ds_visual.list_files[i]} ---")
        result = ds_visual[i] # 调用 __getitem__ 开始分步显示
        
        if result is None:
            # 如果 __getitem__ 返回 None (按 ESC 退出)
            break
        
    cv2.destroyAllWindows()
    print("\n程序结束。")
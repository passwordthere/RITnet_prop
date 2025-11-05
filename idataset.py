import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomPreprocess:
    def __init__(self, target_size=(768, 512)):
        self.target_size = target_size
        self.table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, self.target_size) 
        
        frame = cv2.LUT(frame, self.table.astype('uint8')) 
        frame = self.clahe.apply(np.array(np.uint8(frame)))
        
        # 注意：这里返回的是预处理后的 numpy 数组 (H, W)，类型通常是 np.uint8。
        # 稍后将通过 transforms.ToTensor() 和 transforms.Normalize() 进行进一步处理。
        return frame

inference_transform = transforms.Compose([
    CustomPreprocess(target_size=(768, 512)), # 应用自定义的 cv2 预处理
    transforms.ToTensor(), # 将 (H, W) 的 numpy.ndarray (np.uint8, 0-255) 
                           # 转换为 (C, H, W) 的 torch.float32 Tensor (0.0-1.0)
                           # 对于灰度图像，C=1
    transforms.Normalize([0.5], [0.5]) # 标准化，假设灰度图像只有1个通道
])

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=inference_transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([os.path.join(root_dir, f) 
                                   for f in os.listdir(root_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
        image_tensor = self.transform(image)
            
        return image_tensor, img_path
import cv2
import torch
import numpy as np
import os
import math
import time
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import model_dict


def get_outermost_sclera_point(predict):
    y_coords, x_coords = np.where(predict == 1)
    if len(x_coords) == 0: 
        return None

    outer_corner_x = np.min(x_coords)
    outer_corner_y = y_coords[x_coords == outer_corner_x]
    outer_corner_y = int(np.mean(outer_corner_y))
    return (outer_corner_x, outer_corner_y)


def draw_target_crosshair(frame, predict, frame_id, offset_x=20, offset_y=20):
    """
    在视频帧上绘制一个CS风格的准心，引导操作员。
    
    参数:
    - frame: 原始的BGR视频帧 (来自工业相机)。
    - sclera_mask: 巩膜的二值掩码。
    - offset_x: 目标点相对于基准点的水平偏移。
    - offset_y: 目标点相对于基准点的垂直偏移。
    
    返回:
    - 绘制了准心的视频帧 (副本)。
    """
    frame_with_guide = frame.copy()
    sclera_point = get_outermost_sclera_point(predict)

    if sclera_point is None:
        return frame_with_guide # 预防眨眼

    sclera_x, sclera_y = sclera_point
    target_x = sclera_x - offset_x
    target_y = sclera_y + offset_y 
    
    # 使用正弦函数创建一个在 0 和 1 之间平滑变化的 "脉冲"
    # 调整 0.1 可以改变脉冲的速度
    pulse_factor = (math.sin(frame_id * 0.1) + 1) / 2  # 范围: 0.0 到 1.0
    base_color_bgr = (255, 200, 0) # BGR 格式的青色
    
    # 根据脉冲计算动态颜色
    # 亮度从 50% (128) 变化到 100% (255)
    brightness_min = 0.5
    dynamic_alpha = brightness_min + (1 - brightness_min) * pulse_factor
    
    # 将基础颜色乘以动态亮度
    pulsing_color = (
        int(base_color_bgr[0] * dynamic_alpha),
        int(base_color_bgr[1] * dynamic_alpha),
        int(base_color_bgr[2] * dynamic_alpha)
    )

    # --- 3. 绘制HUD元素 ---
    
    thickness = 1  # 保持线条精细，更具科技感
    
    # (元素 A) 绘制中心目标环 (空心)
    # 也可以让半径随着脉冲变化，但颜色变化更稳定
    radius = 8
    cv2.circle(frame_with_guide, (target_x, target_y), radius, pulsing_color, thickness)

    # (元素 B) 绘制四个角标 (方案二的变体)
    size = 12  # 角标臂长
    gap = 4    # 角标距离中心的间隙
    
    # 左上角
    cv2.line(frame_with_guide, (target_x - gap, target_y - gap), (target_x - gap - size, target_y - gap), pulsing_color, thickness)
    cv2.line(frame_with_guide, (target_x - gap, target_y - gap), (target_x - gap, target_y - gap - size), pulsing_color, thickness)
    
    # 右上角
    cv2.line(frame_with_guide, (target_x + gap, target_y - gap), (target_x + gap + size, target_y - gap), pulsing_color, thickness)
    cv2.line(frame_with_guide, (target_x + gap, target_y - gap), (target_x + gap, target_y - gap - size), pulsing_color, thickness)
    
    # 左下角
    cv2.line(frame_with_guide, (target_x - gap, target_y + gap), (target_x - gap - size, target_y + gap), pulsing_color, thickness)
    cv2.line(frame_with_guide, (target_x - gap, target_y + gap), (target_x - gap, target_y + gap + size), pulsing_color, thickness)

    # 右下角
    cv2.line(frame_with_guide, (target_x + gap, target_y + gap), (target_x + gap + size, target_y + gap), pulsing_color, thickness)
    cv2.line(frame_with_guide, (target_x + gap, target_y + gap), (target_x + gap, target_y + gap + size), pulsing_color, thickness)

    return frame_with_guide


class CustomPreprocess:
    def __init__(self, target_size=(768, 512)):
        self.target_size = target_size
        self.table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        frame = cv2.resize(frame, self.target_size)
        frame_uint8 = cv2.LUT(frame, self.table.astype('uint8'))    # 注意：LUT 操作需要 uint8 输入
        frame_clahe = self.clahe.apply(frame_uint8)                 # clahe.apply 也需要 uint8 输入
        
        return frame_clahe

inference_transform = transforms.Compose([
    CustomPreprocess(target_size=(768, 512)), # 应用自定义的 cv2 预处理
    transforms.ToTensor(), # (H, W) np.uint8 -> (1, H, W) torch.float32
    transforms.Normalize([0.5], [0.5]) # 标准化
])

def get_predictions(output):
    bs,c,h,w = output.size()
    indices = torch.argmax(output, dim=1) # (B, H, W)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices

def run_realtime_inference(model, device, transform):
    IMAGE_DIR = 'mock_camera_od/images/'
    image_files = sorted([
            os.path.join(IMAGE_DIR, f) 
            for f in os.listdir(IMAGE_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    print("开始实时推理... 按 'q' 退出。")

    TARGET_FPS = 10.0
    # 目标每帧时长 (毫秒)
    TARGET_MS_PER_FRAME = int(1000 / TARGET_FPS)
    
    frame_id = 0
    with torch.no_grad():
        for img_path in image_files:
            start_time = time.time()
            frame = cv2.imread(img_path)
            frame_resized = cv2.resize(frame, (768, 512))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            image_tensor = transform(gray_frame)
            data = image_tensor.unsqueeze(0).to(device) # shape: [1, 1, 512, 768]

            output = model(data) # shape: [1, NumClasses, 512, 768]

            predict = get_predictions(output) 
            pred_cpu = predict[0].cpu().numpy()

            display_frame = draw_target_crosshair(frame_resized,  pred_cpu, frame_id,  offset_x=20,  offset_y=20)
            frame_id += 1
            cv2.imshow('real', display_frame)

            # elapsed_time_ms = int((time.time() - start_time) * 1000)
            # wait_time_ms = max(1, TARGET_MS_PER_FRAME - elapsed_time_ms)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


# @profile
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_dict['densenet']
    model = model.to(device)
    filename = 'infrared.pkl'
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    print(f"模型的默认数据类型是: {next(model.parameters()).dtype}")

    # 1. 运行离线数据集
    # print("运行离线数据集...")
    # run_offline_inference(model, device) # 你需要把你的 main() 循环放进这个函数

    # 2. 运行实时摄像头
    print("运行实时推理...")
    run_realtime_inference(model, device, inference_transform)


if __name__ == '__main__':
    
    main()
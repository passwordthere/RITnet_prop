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


def draw_target_crosshair(frame, sclera_mask, frame_id, offset_x=20, offset_y=20):
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
    # 创建一个与原帧相同大小的透明层，用于绘制叠加效果
    overlay = frame.copy()
    output = frame.copy() # 用于最终叠加的副本

    # 1. 找到基准点并计算目标点
    sclera_point = get_outermost_sclera_point(sclera_mask)

    if sclera_point is None:
        return output # 如果没找到目标，返回原始帧 (或使用状态保持)

    sclera_x, sclera_y = sclera_point
    target_x = sclera_x + offset_x
    target_y = sclera_y + offset_y
    
    # --- 2. 动画参数计算 ---
    
    # 基础脉冲 (用于亮度和大小变化)
    pulse_base = (math.sin(frame_id * 0.08) + 1) / 2 # 0.0到1.0，更慢一些
    
    # 扫描波参数
    scan_speed = 0.06 # 调整扫描速度
    scan_offset = frame_id * scan_speed
    scan_progress = scan_offset % (2 * math.pi) # 0 到 2*pi 的循环
    
    # 颜色：从深蓝/青色到亮青色渐变
    base_hue = 180 # 青色的HSV色相
    
    # 动态色相和亮度 (例如，在青色附近小范围波动，并随脉冲变亮)
    hue_variation = 10 * math.sin(frame_id * 0.15) # 色相小范围波动
    current_hue = int((base_hue + hue_variation) % 180) # OpenCV H范围是0-179
    
    # 亮度从低到高，对应脉冲
    current_saturation = 255 # 饱和度保持高
    current_value = int(100 + 155 * pulse_base) # 亮度从100到255
    
    # 将HSV转换回BGR
    hsv_color = np.array([[[current_hue, current_saturation, current_value]]], dtype=np.uint8)
    dynamic_color_bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
    
    # 设置线条粗细
    thickness = 1
    
    # --- 3. 绘制核心引导元素 (类似之前的角标+中心环) ---
    radius = 8
    cv2.circle(overlay, (target_x, target_y), radius, dynamic_color_bgr, thickness)

    size = 12  # 角标臂长
    gap = 4    # 角标距离中心的间隙
    
    # 绘制四个角标
    points = [
        ((target_x - gap, target_y - gap), (target_x - gap - size, target_y - gap), (target_x - gap, target_y - gap - size)), # TL
        ((target_x + gap, target_y - gap), (target_x + gap + size, target_y - gap), (target_x + gap, target_y - gap - size)), # TR
        ((target_x - gap, target_y + gap), (target_x - gap - size, target_y + gap), (target_x - gap, target_y + gap + size)), # BL
        ((target_x + gap, target_y + gap), (target_x + gap + size, target_y + gap), (target_x + gap, target_y + gap + size)), # BR
    ]
    
    for p1, p2, p3 in points:
        cv2.line(overlay, p1, p2, dynamic_color_bgr, thickness)
        cv2.line(overlay, p1, p3, dynamic_color_bgr, thickness)

    # --- 4. 绘制“扫描波”效果 ---
    # 这是一个向外扩散的透明圆环，模拟扫描
    
    max_scan_radius = 50 # 扫描波的最大半径
    current_scan_radius = int(max_scan_radius * (1 - abs(math.sin(scan_progress / 2)))) # 从0到max_radius再到0
    
    # 扫描波的透明度：在扩散时逐渐变淡
    scan_alpha = 0.5 * (1 - (current_scan_radius / max_scan_radius))**2 # 非线性衰减
    scan_alpha = np.clip(scan_alpha, 0.1, 0.5) # 确保不完全透明
    
    # 绘制半透明圆环
    if current_scan_radius > 0:
        cv2.circle(overlay, (target_x, target_y), current_scan_radius, dynamic_color_bgr, 1) # 细线
        # 为了实现透明度，这里需要将overlay混合到output上
        # cv2.addWeighted 是在最后一次性完成，所以这里我们只需要在overlay上画

    # --- 5. 绘制“数据流”指示器 ---
    # 在目标点周围绘制随机分布但有方向性的小点或线段，模拟数据汇聚/流出
    
    num_data_particles = 10
    particle_max_dist = 20 # 粒子最大距离
    particle_speed = 0.5 # 粒子移动速度
    
    for i in range(num_data_particles):
        # 每个粒子的起始角度和距离
        angle = (i * (360 / num_data_particles) + frame_id * particle_speed) % 360
        rad = math.radians(angle)
        
        # 粒子沿着径向稍微来回移动
        current_dist = particle_max_dist * ((math.sin(frame_id * 0.15 + i) + 1) / 2) # 每个粒子有不同的微动
        
        # 粒子的起点和终点，模拟短线段
        start_x = int(target_x + (radius + current_dist) * math.cos(rad))
        start_y = int(target_y + (radius + current_dist) * math.sin(rad))
        
        end_x = int(target_x + (radius + current_dist + 3) * math.cos(rad)) # 短线段长度3
        end_y = int(target_y + (radius + current_dist + 3) * math.sin(rad))
        
        # 粒子的透明度也随距离和脉冲变化
        particle_alpha = 0.2 + 0.3 * pulse_base # 0.2到0.5的透明度
        particle_color_bgr = (
            int(dynamic_color_bgr[0] * particle_alpha * 2),
            int(dynamic_color_bgr[1] * particle_alpha * 2),
            int(dynamic_color_bgr[2] * particle_alpha * 2)
        )
        
        cv2.line(overlay, (start_x, start_y), (end_x, end_y), particle_color_bgr, 1)
        
    # --- 6. 将叠加层与原始帧混合 ---
    # 这里使用一个固定的alpha值来控制整个HUD的可见性，
    # 也可以让这个alpha值随着脉冲变化，使其整体“呼吸”
    overall_alpha = 0.7 + 0.3 * pulse_base # 整体透明度从70%到100%
    cv2.addWeighted(overlay, overall_alpha, frame, 1 - overall_alpha, 0, output)

    return output


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
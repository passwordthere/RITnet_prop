import cv2
import torch
import numpy as np
import os
import math
import time
import socket
import json
from multiprocessing import shared_memory 
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


IMG_HEIGHT = 512
IMG_WIDTH = 768
IMG_CHANNELS = 3
SHM_SIZE = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS
SHM_NAME = "eye_tracker_shm"
UDS_NAME = "/tmp/eye_tracker_uds"


def run_realtime_inference(model, device, transform, shm, uds_socket):
    IMAGE_DIR = 'mock_camera_od/images/'
    image_files = sorted([
            os.path.join(IMAGE_DIR, f) 
            for f in os.listdir(IMAGE_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    print("开始实时推理... C++ GUI 应该会显示图像。")

    # 创建一个 numpy 数组，其缓冲区指向共享内存
    shm_array = np.ndarray((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8, buffer=shm.buf)

    TARGET_FPS = 10.0
    TARGET_MS_PER_FRAME = 1.0 / TARGET_FPS
    
    frame_id = 0
    with torch.no_grad():
        for img_path in image_files:
            loop_start_time = time.time()
            frame = cv2.imread(img_path)
            frame_resized = cv2.resize(frame, (768, 512))
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            image_tensor = transform(gray_frame)
            data = image_tensor.unsqueeze(0).to(device) # shape: [1, 1, 512, 768]

            output = model(data) # shape: [1, NumClasses, 512, 768]

            predict = get_predictions(output) 
            pred_cpu = predict[0].cpu().numpy()

            display_frame = draw_target_crosshair(frame_resized,  pred_cpu, frame_id,  offset_x=20,  offset_y=20)
            
            # --- 关键步骤: 写入共享内存并发送信号 ---
            try:
                # 1. 将处理后的帧(BGR)复制到共享内存
                shm_array[:] = display_frame

                # 2. 准备元数据信号
                metadata = {
                    "frame_id": frame_id,
                    "timestamp": time.time()
                }
                # 确保以换行符结束，以便 C++ 按行读取
                signal_data = (json.dumps(metadata) + '\n').encode('utf-8')

                # 3. 通过 UDS 发送信号
                uds_socket.sendall(signal_data)

            except (ConnectionResetError, BrokenPipeError):
                print("C++ GUI 已断开连接。正在退出...")
                break
            except Exception as e:
                print(f"发送信号时出错: {e}")
                break
            # --- 结束关键步骤 ---

            frame_id += 1
            
            # 简单的 FPS 控制 (模拟相机)
            elapsed_time = time.time() - loop_start_time
            wait_time = max(0, TARGET_MS_PER_FRAME - elapsed_time)
            time.sleep(wait_time)

    print("模拟图像处理完毕。")


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

    shm = None
    uds_socket = None
    
    try:
        # --- 1. 初始化共享内存 ---
        try:
            # 尝试创建新的共享内存块
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
            print(f"创建了共享内存: {SHM_NAME}")
        except FileExistsError:
            # 如果已存在 (上次崩溃导致)，先连接再销毁，然后再创建
            print("警告: 发现陈旧的共享内存块。正在清理...")
            temp_shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            temp_shm.close()
            temp_shm.unlink() # 销毁它
            # 再次创建
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
            print(f"重新创建了共享内存: {SHM_NAME}")

        # --- 2. 初始化 UDS 客户端套接字 ---
        uds_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        print(f"正在连接到 C++ UDS 服务器: {UDS_NAME}...")
        while True:
            try:
                uds_socket.connect(UDS_NAME)
                print("已连接到 C++ 服务器!")
                break
            except (ConnectionRefusedError, FileNotFoundError):
                print("等待 C++ 服务器启动...")
                time.sleep(1)
            except KeyboardInterrupt:
                raise # 允许 Ctrl+C 退出

        # --- 3. 运行主循环 ---
        print("运行实时推理...")
        run_realtime_inference(model, device, inference_transform, shm, uds_socket)

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在关闭...")
    except Exception as e:
        print(f"发生未处理的异常: {e}")
    finally:
        # --- 4. 清理 ---
        if uds_socket:
            uds_socket.close()
            print("UDS 套接字已关闭")
        if shm:
            shm.close()   # 关闭此进程的句柄
            shm.unlink()  # 销毁共享内存块
            print("共享内存已销毁")
        print("Python 脚本退出。")


if __name__ == '__main__':
    
    main()
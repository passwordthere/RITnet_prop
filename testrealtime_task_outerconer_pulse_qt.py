import cv2
import torch
import numpy as np
import os
import math
import time
import socket
import struct
import threading
from torchvision import transforms
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

# --- 1. 定义线程安全的全局变量 ---
global_latest_clean_frame = None
global_frame_lock = threading.Lock() # 锁，用于保护上面的变量

# --- 2. 新的 Python 协议常量 ---
TYPE_DISPLAY = 0x01
TYPE_SNAPSHOT = 0x02

# --- 3. [新] 监听 C++ 请求的子线程 ---
def request_listener(sock: socket.socket):
    """
    此函数在一个单独的线程中运行。
    它只负责监听 C++ 发回的请求。
    """
    global global_latest_clean_frame
    global global_frame_lock
    
    try:
        while True:
            # recv(1) 是阻塞的，但没关系，因为它在自己的线程里
            # 它会一直等到 C++ 发来数据 (比如 'S')
            data = sock.recv(1) 
            
            if not data:
                print("[Listener] C++ Socket 已关闭。")
                break # C++ 断开连接

            if data == b'S':
                print("[Listener] 收到 'S' 快照请求！")
                
                # 抓取最新的原图
                frame_to_send = None
                with global_frame_lock:
                    if global_latest_clean_frame is not None:
                        frame_to_send = global_latest_clean_frame.copy()
                
                if frame_to_send is not None:
                    # 编码并发送原图
                    result_clean, encoded_clean = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if result_clean:
                        data_bytes = encoded_clean.tobytes()
                        # [协议] 发送 TYPE_SNAPSHOT (0x02) + Size + Data
                        header_type = struct.pack('!B', TYPE_SNAPSHOT)
                        header_size = struct.pack('!I', len(data_bytes))
                        sock.sendall(header_type + header_size + data_bytes)
                        print(f"[Listener] 已发送 {len(data_bytes)} 字节的原图。")
                    else:
                        print("[Listener] 原图编码失败。")
                else:
                    print("[Listener] 收到请求，但没有可用的原图。")

    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"[Listener] 线程连接中断: {e}")
    finally:
        print("[Listener] 监听线程退出。")


# --- 4. [修改] 主推理循环 (在主线程运行) ---
def run_realtime_inference(model, device, transform, sock: socket.socket):
    """
    这是主循环，只负责跑推理和发送实时画面。
    """
    global global_latest_clean_frame
    global global_frame_lock

    IMAGE_DIR = 'mock_camera_od/images/'
    image_files = sorted([
            os.path.join(IMAGE_DIR, f) 
            for f in os.listdir(IMAGE_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    if not image_files:
        print(f"错误：在 {IMAGE_DIR} 中没有找到图片。")
        return

    print("[MainLoop] 开始实时推理...")
    frame_id = 0
    with torch.no_grad():
        while True:  # 无限循环
            for img_path in image_files:
                frame = cv2.imread(img_path)
                frame_resized = cv2.resize(frame, (768, 512)) # "原图"
                
                # --- [线程安全] 更新全局原图 ---
                with global_frame_lock:
                    global_latest_clean_frame = frame_resized
                # ----------------------------------
                
                gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                image_tensor = transform(gray_frame)
                data = image_tensor.unsqueeze(0).to(device) 

                output = model(data) 
                predict = get_predictions(output) 
                pred_cpu = predict[0].cpu().numpy()

                display_frame = draw_target_crosshair(frame_resized,  pred_cpu, frame_id,  offset_x=20,  offset_y=20)
                frame_id += 1

                # 编码并发送 "带准星" 的图
                result_display, encoded_display = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # 降低一点质量以提高帧率
                
                if not result_display:
                    print("[MainLoop] 准星图编码失败")
                    continue

                data_bytes_display = encoded_display.tobytes()

                try:
                    # [协议] 发送 TYPE_DISPLAY (0x01) + Size + Data
                    header_type = struct.pack('!B', TYPE_DISPLAY)
                    header_size = struct.pack('!I', len(data_bytes_display))
                    sock.sendall(header_type + header_size + data_bytes_display)
                
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
                    print(f"[MainLoop] 连接中断 (C++ 窗口可能已关闭): {e}。退出...")
                    return # 退出主循环


def init_model(onnx=False):
    if onnx:
        model_path = 'log/weights_infrared_best.onnx'
        os.makedirs(trt_cache_path, exist_ok=True)
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': trt_cache_path,
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0,
            }),
            'CPUExecutionProvider'
        ]
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Model loaded! Provider: {session.get_providers()}")
        return session, input_name, output_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_dict['densenet']
    model = model.to(device)
    filename = 'infrared.pkl'
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    print(f"模型的默认数据类型是: {next(model.parameters()).dtype}")    

    return model


# --- 5. [修改] main 函数 ---
# @profile
def main():
    model = init_model(onnx=False)

    # --- [修改] Socket 连接逻辑 ---
    try:
        print("正在连接到 Qt 接收器 (localhost:12345)...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 12345))
        print("已连接！")
    except ConnectionRefusedError:
        print("连接失败。请确保 C++ Qt 应用程序正在运行。")
        return

    # --- [修改] 启动多线程 ---
    # 1. 创建监听线程
    listener = threading.Thread(target=request_listener, args=(sock,), daemon=True) # daemon=True 表示主线程退出时, 该线程也退出
    # 2. 启动监听线程
    listener.start()
    
    # 3. 在主线程中运行推理循环
    try:
        run_realtime_inference(model, device, inference_transform, sock)
    except KeyboardInterrupt:
        print("用户请求退出...")
    finally:
        # 4. 清理
        print("关闭 Socket...")
        sock.close()
        print("程序退出。")


if __name__ == '__main__':
    
    main()
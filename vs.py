import threading
import queue
import time
import cv2
import numpy as np
import sys
import torch

from models import model_dict
# from models import model_dict # 假设已导入

# --- 全局/外部变量定义 ---
# 必须先定义，供 load_inference_model 和 NEURAL_NETWORK_INFERENCE 使用

def load_inference_model(model_name='densenet', filename='best_model.pkl'):
    """加载、配置并返回用于推理的 PyTorch 模型。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化模型 (替换为 model_dict[model_name])
    model = model_dict['densenet']
    model = model.to(device)
    
    # 尝试加载权重
    try:
        state_dict = torch.load(filename, map_location=device) 
        model.load_state_dict(state_dict)
        print(f"✅ 模型权重从 '{filename}' 加载成功。")
    except FileNotFoundError:
        print(f"⚠️ 警告: 模型文件 '{filename}' 未找到，使用未训练的模型。")

    model.eval() 
    return model, device

# 初始化全局模型和设备
INFERENCE_MODEL, DEVICE_GLOBAL = load_inference_model(filename='best_model.pkl')
print(f"模型默认数据类型: {next(INFERENCE_MODEL.parameters()).dtype}")

# 预处理常量
MEAN_GPU = torch.tensor([0.5], dtype=torch.float32, device=DEVICE_GLOBAL).view(1, 1, 1, 1)
STD_GPU = torch.tensor([0.5], dtype=torch.float32, device=DEVICE_GLOBAL).view(1, 1, 1, 1) 

# --- 核心操作函数 (包含实现) ---

def PREPROCESS_ON_GPU(input_data):
    # 实现标准化 (x - mean) / std
    return (input_data - MEAN_GPU) / STD_GPU

def NEURAL_NETWORK_INFERENCE(processed_data):
    """
    【实现】执行神经网络推理。
    输入：预处理后的 Tensor (在 GPU)。
    输出：模型输出 Tensor (在 GPU)。
    """
    # 无需 global INFERENCE_MODEL, 因为它是只读操作
    with torch.no_grad():
        # 执行前向传播
        model_output = INFERENCE_MODEL(processed_data)
        
    return model_output

def POST_PROCESS_ON_GPU(model_output, original_frame):
    # 假设这里包含了原始代码中的 get_predictions(output) 和结果拼接
    
    # 模拟结果：对模型输出进行逆归一化并转换为 NumPy
    display_frame_cpu = torch.clamp((model_output * 0.5 + 0.5) * 255.0, 0, 255).byte().cpu().squeeze().numpy()
    
    if display_frame_cpu.ndim == 3 and display_frame_cpu.shape[0] == 1:
        display_frame_cpu = display_frame_cpu.squeeze(0) # 移除 C=1 维度
        
    return display_frame_cpu

# --- 线程和主循环 (与之前保持一致) ---
image_queue = queue.Queue(maxsize=1) 
is_running = True 

def display_worker(q: queue.Queue):
    global is_running
    print("【显示线程】启动。等待图像数据...")
    
    while is_running:
        try:
            frame = q.get(timeout=0.05) 
            cv2.imshow("Asynchronous Real-time Output (Press 'q' to Quit)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running = False
                break
                
        except queue.Empty:
            continue

    cv2.destroyAllWindows()
    print("【显示线程】退出。")


def gpu_inference_simulator(q: queue.Queue):
    global is_running
    frame_count = 0
    print(f"【主 GPU 线程】启动。目标设备: {DEVICE_GLOBAL}")

    while is_running:
        start_time = time.time()
        
        # 模拟一批数据 [B, C, H, W] = [1, 1, 300, 500]
        # 假设这里是已完成 ToTensor 缩放至 [0.0, 1.0] 的数据
        simulated_input_frame = torch.rand((1, 1, 300, 500), dtype=torch.float32, device=DEVICE_GLOBAL)
        
        # 1. 预处理 (在 GPU 上进行)
        data_gpu = PREPROCESS_ON_GPU(simulated_input_frame)
        
        # 2. 神经网络推理 (在 GPU 上进行)
        model_output_gpu = NEURAL_NETWORK_INFERENCE(data_gpu) # <--- 执行推理
        
        # 3. 后处理 (在 GPU 上进行)
        display_frame_cpu = POST_PROCESS_ON_GPU(model_output_gpu, simulated_input_frame)
        
        # 4. 标记和数据准备
        current_fps = 1.0 / (time.time() - start_time)
        # 确保 cv2.putText 接受 3D BGR 或 2D 灰度图
        cv2.putText(display_frame_cpu, f"GPU Inference Frame: {frame_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(display_frame_cpu, f"FPS: {current_fps:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # 5. 将最新结果放入队列
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass 
                
        q.put(display_frame_cpu)
        
        end_time = time.time()
        
        print(f"【主 GPU 线程】处理帧 {frame_count} 耗时: {(end_time - start_time)*1000:.2f} ms")
        frame_count += 1
        
    print("【主 GPU 线程】退出。")


if __name__ == '__main__':
    # 启动显示工作线程
    display_thread = threading.Thread(target=display_worker, args=(image_queue,))
    display_thread.daemon = True 
    display_thread.start()
    
    try:
        gpu_inference_simulator(image_queue)
    except KeyboardInterrupt:
        print("\n捕获到中断信号。正在关闭...")
    
    # 退出清理
    is_running = False
    display_thread.join() 
    cv2.destroyAllWindows()
    print("程序完全退出。")
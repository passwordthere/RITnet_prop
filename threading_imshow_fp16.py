import pathlib
import threading
import queue
import time
import cv2
import numpy as np
import sys
import torch
import itertools

# 假设 models 模块和 model_dict 保持不变
from models import model_dict


def load_inference_model(model_name='densenet', filename='best_model.pkl'):
    # 确保使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict['densenet']
    
    # *** 1. 核心修改：将模型转换为 FP16 ***
    # 注意：某些操作（如 BatchNorm）在 FP16 下可能不稳定，
    # 更好的做法是使用 torch.amp 自动混合精度，
    # 但此处为实现纯 FP16 推理，直接使用 .half()
    model = model.half() 
    
    model = model.to(device)
    # 加载模型时，需要指定 map_location=device 以确保在正确设备上
    # 同时如果存储的权重是 FP32，加载时 PyTorch 会自动进行类型转换。
    state_dict = torch.load(filename, map_location=device) 
    model.load_state_dict(state_dict)
    model.eval() 
    return model, device
    
INFERENCE_MODEL, DEVICE_GLOBAL = load_inference_model(filename='best_model.pkl')
print(f"模型默认数据类型: {next(INFERENCE_MODEL.parameters()).dtype}")


def PREPROCESS_ON_GPU(input_data, target_size=(400, 640)):
    """
    在 GPU 上执行预处理。
    输入：原始 NumPy 数组。
    输出：预处理后的 FP16 Tensor (仍在 GPU)。
    """
    
    # *** 2. 核心修改：将常量转换为 FP16 (torch.float16) ***
    MEAN_GPU = torch.tensor([0.5], dtype=torch.float16, device=DEVICE_GLOBAL).view(1, 1, 1) 
    STD_GPU = torch.tensor([0.5], dtype=torch.float16, device=DEVICE_GLOBAL).view(1, 1, 1)

    LUT_TABLE_CPU = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    # LUT 表本身使用 float32，因为索引（lut_input_gpu）是 long 类型
    LUT_TABLE_GPU = torch.tensor(LUT_TABLE_CPU, dtype=torch.float32, device=DEVICE_GLOBAL)

    # 步骤 A: 数据传输 (CPU -> GPU)
    # 输入帧使用 FP16 精度
    frame_gpu = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0).to(DEVICE_GLOBAL, dtype=torch.float16)
    frame_gpu = frame_gpu / 255.0
    
    # 步骤 B: 图像缩放
    # F.interpolate 最好在 FP32 上执行以避免精度问题，然后再转回 FP16
    resized_gpu_fp32 = torch.nn.functional.interpolate(
        frame_gpu.float(), # 临时转为 FP32 进行插值
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    resized_gpu = resized_gpu_fp32.half() # 插值后转回 FP16
    
    # 步骤 C: LUT 查找表 (模拟 cv2.LUT)
    # LUT 查找表的输入必须是 long 类型，查找后结果转回 FP16
    lut_input_gpu = (resized_gpu * 255.0).clamp(0, 255).long()
    
    # LUT 查找结果是 FP32，需要转回 FP16
    lut_output_gpu = LUT_TABLE_GPU[lut_input_gpu].squeeze(0).squeeze(-1).half() 
    lut_output_gpu = lut_output_gpu / 255.0
    
    # 步骤 D: 归一化 (使用 FP16 常量)
    final_data_gpu = lut_output_gpu.unsqueeze(0)
    preprocessed_data_gpu = (final_data_gpu - MEAN_GPU) / STD_GPU
    
    # 确保输出是 FP16
    return preprocessed_data_gpu


def NEURAL_NETWORK_INFERENCE(processed_data):
    """
    执行神经网络推理。
    输入：预处理后的 FP16 Tensor (在 GPU)。
    输出：模型输出 FP16 Tensor (在 GPU)。
    """
    # 确保模型在 eval 模式
    with torch.no_grad(): 
        model_output = INFERENCE_MODEL(processed_data)
    # 模型输出也应该是 FP16
    return model_output

def POST_PROCESS_ON_GPU(output, input):
    """
    在 GPU 上执行后处理。
    输入：模型输出 Tensor (FP16) 和输入 Tensor (FP16)。
    输出：最终的显示图像 (NumPy 数组, 在 CPU)。
    """
    
    # 为了避免 FP16 在后处理中的精度问题，建议临时转回 FP32 进行后处理。
    output = output.float()
    input = input.float()

    bs,c,h,w = output.size()
    values, indices = output.max(1)
    indices = indices.view(bs,h,w)

    inp_gpu = input.squeeze() * 0.5 + 0.5 
    img_orig_gpu = torch.clamp(inp_gpu, 0, 1) # [0, 1] 范围
    pred_gpu = indices.squeeze() / 3.0 # [0, 1] 范围
    combine_gpu = torch.hstack([img_orig_gpu, pred_gpu]) 
    display_frame_cpu = (combine_gpu * 255.0).byte().cpu().numpy() 
    return display_frame_cpu


# 1. 设置线程安全的队列
image_queue = queue.Queue(maxsize=1) 
is_running = True 

# 以下函数 (display_worker, get_image_paths, gpu_inference_simulator) 
# 保持与原代码相同，它们负责线程调度、图像加载和计时，不需要修改。

def display_worker(q: queue.Queue):
    """
    负责从队列中取出图像并使用 cv2.imshow 进行显示的独立工作线程。
    """
    global is_running
    print("【显示线程】启动。等待图像数据...")

    frame_count_disp = 0
    start_time_disp = time.time()
    display_fps = 0
    
    while is_running:
        try:
            frame = q.get(timeout=0.05) 
            cv2.imshow("Asynchronous Real-time Output (Press 'q' to Quit)", frame)

            frame_count_disp += 1
            elapsed_time = time.time() - start_time_disp
            
            if elapsed_time > 1.0:
                display_fps = frame_count_disp / elapsed_time
                print(f"【显示线程】实际显示 FPS: {display_fps:.2f}")
                frame_count_disp = 0
                start_time_disp = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running = False
                break
                
        except queue.Empty:
            continue

    cv2.destroyAllWindows()
    print("【显示线程】退出。")


def get_image_paths(base_dir='dataset/test/images'):
    base_dir = pathlib.Path(base_dir)
    path_generator = base_dir.glob('*_od.jpg')
    paths = [str(p) for p in path_generator]
    paths.sort()
    return paths

IMAGE_PATHS = get_image_paths()


def gpu_inference_simulator(q: queue.Queue):
    """
    模拟 GPU 推理的主线程。
    """
    global is_running
    frame_count = 0
    print(f"【主 GPU 线程】启动。目标设备: {DEVICE_GLOBAL}")

    path_iterator = itertools.cycle(IMAGE_PATHS)
    while is_running:
        # **关键：使用 CUDA 事件进行精确计时和同步**
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 记录 I/O 和 CPU 预处理的开始时间
        cpu_start_time = time.time()
        
        current_path = next(path_iterator)
        frame = cv2.imread(current_path, 0)
        frame = np.expand_dims(frame, axis=-1)
        
        # 开始记录 GPU 时间
        start_event.record()

        # --- 1. 预处理 (在 GPU 上进行) ---
        data_gpu = PREPROCESS_ON_GPU(frame)
        
        # --- 2. 神经网络推理 (在 GPU 上进行) ---
        model_output_gpu = NEURAL_NETWORK_INFERENCE(data_gpu)
        
        # --- 3. 后处理 (在 GPU 上进行) ---
        display_frame_cpu = POST_PROCESS_ON_GPU(model_output_gpu, data_gpu)

        # 停止记录 GPU 时间
        end_event.record()
        
        # 等待 GPU 完成所有操作 (同步)
        torch.cuda.synchronize() 
        
        # 获取纯 GPU 计算时间 (毫秒)
        pure_gpu_time_ms = start_event.elapsed_time(end_event) 
        
        # 获取总端到端时间
        end_time_total = time.time()
        total_latency_ms = (end_time_total - cpu_start_time) * 1000
        
        # 4. 打印和标记性能数据
        pure_gpu_fps = 1000.0 / pure_gpu_time_ms
        
        cv2.putText(display_frame_cpu, f"GPU FPS (Pure): {pure_gpu_fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame_cpu, f"Total Latency: {total_latency_ms:.2f}ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 5. 将最新结果放入队列
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass 
                
        q.put(display_frame_cpu)
        
        print(f"【主 GPU 线程】处理帧 {frame_count} | 纯 GPU 耗时: {pure_gpu_time_ms:.2f} ms | 总耗时: {total_latency_ms:.2f} ms")
        frame_count += 1
        
    print("【主 GPU 线程】退出。")


if __name__ == '__main__':
    # 预热：首次运行时会进行 CUDA Kernel 编译，所以需要运行几帧让性能稳定。
    print("--- 正在预热 GPU (运行 5 帧) ---")
    
    # 预热模型和输入数据
    dummy_frame = np.zeros((400, 640, 1), dtype=np.uint8)
    for _ in range(5):
        with torch.no_grad():
            dummy_input = PREPROCESS_ON_GPU(dummy_frame)
            _ = INFERENCE_MODEL(dummy_input)
    torch.cuda.synchronize()
    print("--- 预热完成。开始正式推理 ---")

    display_thread = threading.Thread(target=display_worker, args=(image_queue,))
    display_thread.daemon = True 
    display_thread.start()
    
    # 启动主 GPU 模拟线程
    try:
        gpu_inference_simulator(image_queue)
    except KeyboardInterrupt:
        print("\n捕获到中断信号。正在关闭...")
    
    is_running = False
    display_thread.join()
    cv2.destroyAllWindows()
    print("程序完全退出。")
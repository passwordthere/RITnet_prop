import pathlib
import threading
import queue
import time
import cv2
import numpy as np
import sys
import torch
import itertools

from models import model_dict


def load_inference_model(model_name='densenet', filename='best_model.pkl'):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model = model_dict['densenet']
    model = model.to(device)
    state_dict = torch.load(filename, map_location=device) 
    model.load_state_dict(state_dict)
    model.eval() 
    return model, device
    
INFERENCE_MODEL, DEVICE_GLOBAL = load_inference_model(filename='best_model.pkl')
print(f"模型默认数据类型: {next(INFERENCE_MODEL.parameters()).dtype}")


def PREPROCESS_ON_GPU(input_data, target_size=(400, 640)):
    """
    占位符函数：在 GPU 上执行预处理。
    输入：原始 Tensor (已移至 GPU)。
    输出：预处理后的 Tensor (仍在 GPU)。
    例如：归一化 (x - mean) / std，通道排序等。
    """

    MEAN_GPU = torch.tensor([0.5], dtype=torch.float32, device=DEVICE_GLOBAL).view(1, 1, 1) 
    STD_GPU = torch.tensor([0.5], dtype=torch.float32, device=DEVICE_GLOBAL).view(1, 1, 1)

    LUT_TABLE_CPU = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    LUT_TABLE_GPU = torch.tensor(LUT_TABLE_CPU, dtype=torch.float32, device=DEVICE_GLOBAL)

    # 步骤 A: 数据传输 (CPU -> GPU)
    frame_gpu = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0).to(DEVICE_GLOBAL, dtype=torch.float32)
    frame_gpu = frame_gpu / 255.0
    # 步骤 B: 图像缩放
    resized_gpu = torch.nn.functional.interpolate(
        frame_gpu, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    # 步骤 C: LUT 查找表 (模拟 cv2.LUT)
    lut_input_gpu = (resized_gpu * 255.0).clamp(0, 255).long()
    lut_output_gpu = LUT_TABLE_GPU[lut_input_gpu].squeeze(0).squeeze(-1)
    lut_output_gpu = lut_output_gpu / 255.0
    # 步骤 D: CLAHE (自适应直方图均衡化)
    # clahe_output_gpu = ke.equalize_clahe(lut_output_gpu.unsqueeze(0).unsqueeze(0), clip_limit=1.5, grid_size=(8, 8))
    final_data_gpu = lut_output_gpu.unsqueeze(0)

    preprocessed_data_gpu = (final_data_gpu - MEAN_GPU) / STD_GPU
    return preprocessed_data_gpu

def NEURAL_NETWORK_INFERENCE(processed_data):
    """
    占位符函数：执行神经网络推理。
    输入：预处理后的 Tensor (在 GPU)。
    输出：模型输出 Tensor (在 GPU)。
    例如：output = model(processed_data)
    """
    # time.sleep(0.07)
    with torch.no_grad(): model_output = INFERENCE_MODEL(processed_data)
    return model_output

def POST_PROCESS_ON_GPU(output, input):
    """
    在 GPU 上执行后处理。
    输入：模型输出 Tensor (GPU) 和原始 NumPy 数组 (CPU)。
    输出：最终的显示图像 (NumPy 数组, 在 CPU)。
    """

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
# maxsize=1 是关键！这确保队列中永远只保留“最新”的一帧图像。
# 如果显示线程处理得慢，旧的帧会被丢弃，保证实时性（但牺牲了少量帧）。
image_queue = queue.Queue(maxsize=1) 
is_running = True # 用于控制两个线程的退出

def display_worker(q: queue.Queue):
    """
    负责从队列中取出图像并使用 cv2.imshow 进行显示的独立工作线程。
    这是一个 CPU 阻塞操作。
    """
    global is_running
    print("【显示线程】启动。等待图像数据...")

    # --- FPS 实时计算变量 ---
    frame_count_disp = 0
    start_time_disp = time.time()
    display_fps = 0
    # -------------------------
    
    while is_running:
        try:
            # 尝试取出队列中的最新图像，设置 timeout 确保线程可以被中断
            frame = q.get(timeout=0.05) 
            
            # 核心显示操作 (CPU 阻塞)
            cv2.imshow("Asynchronous Real-time Output (Press 'q' to Quit)", frame)

            # --- 实时 FPS 计算 ---
            frame_count_disp += 1
            elapsed_time = time.time() - start_time_disp
            
            # 每隔 1 秒或 30 帧更新一次 FPS
            if elapsed_time > 1.0:
                display_fps = frame_count_disp / elapsed_time
                print(f"【显示线程】实际显示 FPS: {display_fps:.2f}")
                frame_count_disp = 0
                start_time_disp = time.time()
            
            # cv2.waitKey(1) 必须在主线程或显示线程中被调用
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running = False
                break
                
        except queue.Empty:
            # 队列为空时，继续循环，等待新数据
            continue

    cv2.destroyAllWindows()
    print("【显示线程】退出。")


def get_image_paths(base_dir='dataset/test/images'):
    base_dir = pathlib.Path(base_dir)
    path_generator = base_dir.glob('*_od.jpg')
    paths = [str(p) for p in path_generator]
    paths.sort()
    return paths

# 在主线程中调用，获取图片路径列表
IMAGE_PATHS = get_image_paths()


def gpu_inference_simulator(q: queue.Queue):
    """
    模拟 GPU 推理的主线程。
    它负责所有 GPU 操作，并快速将结果扔进队列。
    """
    global is_running
    frame_count = 0
    print(f"【主 GPU 线程】启动。目标设备: {DEVICE_GLOBAL}")

    path_iterator = itertools.cycle(IMAGE_PATHS)
    while is_running:
        start_time = time.time()
        
        current_path = next(path_iterator)
        frame = cv2.imread(current_path, 0)
        frame = np.expand_dims(frame, axis=-1)
        
        # --- 1. 预处理 (在 GPU 上进行) ---
        data_gpu = PREPROCESS_ON_GPU(frame)
        
        # --- 2. 神经网络推理 (在 GPU 上进行) ---
        model_output_gpu = NEURAL_NETWORK_INFERENCE(data_gpu)
        
        # --- 3. 后处理 (在 GPU 上进行) ---
        # **在 GPU 上完成所有 Tensor 操作后，才执行 .cpu().numpy()**
        display_frame_cpu = POST_PROCESS_ON_GPU(model_output_gpu, data_gpu)

        # 4. 模拟结果回传 CPU 后的标记和数据准备
        end_time_inference = time.time()
        current_fps_inference = 1.0 / (end_time_inference - start_time)
        
        cv2.putText(display_frame_cpu, f"GPU Throughput FPS: {current_fps_inference:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame_cpu, f"Latency: {(end_time_inference - start_time)*1000:.2f}ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 5. 将最新结果放入队列
        if q.full():
            try:
                q.get_nowait() # 丢弃旧的，保证实时性
            except queue.Empty:
                pass 
                
        q.put(display_frame_cpu)
        
        end_time = time.time()
        
        # 打印主线程性能 (应该接近纯推理时间)
        print(f"【主 GPU 线程】处理帧 {frame_count} 耗时: {(end_time - start_time)*1000:.2f} ms")
        frame_count += 1
        
    print("【主 GPU 线程】退出。")


if __name__ == '__main__':
    display_thread = threading.Thread(target=display_worker, args=(image_queue,))
    display_thread.daemon = True # 设置为守护线程，主程序退出时自动终止
    display_thread.start()
    
    # 启动主 GPU 模拟线程
    try:
        gpu_inference_simulator(image_queue)
    except KeyboardInterrupt:
        # 允许用户通过 Ctrl+C 退出
        print("\n捕获到中断信号。正在关闭...")
    
    is_running = False
    display_thread.join() # 等待显示线程完全退出
    cv2.destroyAllWindows()
    print("程序完全退出。")
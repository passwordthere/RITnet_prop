import cv2.cuda


def init_model(redness=False):
    """
    Loads the ONNX model and builds/loads the TensorRT engine.
    Initializes GPU preprocessing resources.
    """
    print("Loading model and building/loading TensorRT engine...")
    # ... (ORT Session 加载部分保持不变) ...

    # ----------------------------------------------------
    # 新增：初始化 GPU 预处理资源
    # ----------------------------------------------------
    
    # 1. CLAHE (CUDA) - 可以在 GPU 上执行，但 OpenCV CUDA 模块没有直接的 CLAHE 实现。
    #    这里先跳过 CLAHE 的 CUDA 优化，专注于 cvtColor 和 resize。
    #    如果 CLAHE 仍是瓶颈，需要使用 NPP 或自定义 CUDA Kernel。
    
    # 2. 创建用于颜色/灰度转换的 GpuMat
    #    目标尺寸 W=768, H=512
    gpu_clahe_output = cv2.cuda_GpuMat(512, 768, cv2.CV_8UC1) 
    
    # 3. 创建用于最终模型输入的 GpuMat (Float32, 768x512)
    #    注意：ONNX Runtime/TensorRT 的输入数据需要是 FP32 格式。
    gpu_input_float = cv2.cuda_GpuMat(512, 768, cv2.CV_32FC1)

    # 4. 创建用于 resize 的对象
    gpu_resize = cv2.cuda.createCudaResize(None, cv2.INTER_LINEAR)

    print(f"Model loaded! Provider: {session.get_providers()}")
    return session, input_name, output_name, gpu_clahe_output, gpu_input_float, gpu_resize


def run_segmentation(frame, session, input_name, output_name, gpu_clahe_output, gpu_input_float, gpu_resize):
    
    # --------------------------------------------------
    # 1. Preprocessing (GPU Accelerated)
    # --------------------------------------------------
    
    # 将 CPU 端的 BGR 帧上传到 GPU
    # GpuMat 的创建和上传是 I/O 优化的第一步
    gpu_frame_bgr = cv2.cuda_GpuMat(frame)
    
    # a. BGR to Gray (GPU)
    gpu_frame_gray = cv2.cuda.GpuMat()
    cv2.cuda.cvtColor(gpu_frame_bgr, cv2.COLOR_BGR2GRAY, gpu_frame_gray)

    # b. Resize (GPU)
    # 目标尺寸 (768, 512)
    gpu_frame_gray_resized = gpu_resize.resize(gpu_frame_gray, (768, 512)) 

    # c. Tone mapping/Gamma Correction (GPU)
    #    LUT 和 CLAHE 都是 CPU 操作。我们先尝试仅用 Resize 和 Normalize 加速
    #    如果需要，这部分应被替换为 NPP 或自定义 CUDA Kernel
    
    # d. CLAHE (CPU - **临时回退**) - 这部分需要优化
    frame_gray_resized_cpu = gpu_frame_gray_resized.download()
    
    # (如果 CLAHE 是瓶颈，您需要为 Jetson 找到 CLAHE 的 CUDA/NPP 实现)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    frame_gray_c = clahe.apply(np.uint8(frame_gray_resized_cpu)) # CPU 上的 numpy array

    # --------------------------------------------------
    # 2. Normalize and shape for the model
    # --------------------------------------------------

    # 将处理后的 CPU 数据重新上传到 GPU（这是临时的性能损失点）
    gpu_frame_c = cv2.cuda_GpuMat(frame_gray_c)

    # 归一化 (GPU)
    # 步骤 i: Convert to Float32 and Normalize to [0, 1]
    gpu_frame_c.convertTo(gpu_input_float, cv2.CV_32FC1, 1/255.0)

    # 步骤 ii: Normalize to [-1, 1] -> (data - 0.5) / 0.5
    # 我们需要一个自定义 CUDA Kernel 或使用 cv2.cuda.subtract/divide
    # 这里我们使用简单的方法：下载到 CPU numpy，进行简单的 [-1, 1] 归一化，再上传。
    # *** 为了 ORT/TensorRT 优化，强烈建议使用 TensorRT 预处理插件或自定义 CUDA Kernel 执行此归一化。***

    # 临时使用 CPU 归一化 (需要优化)
    data = gpu_input_float.download()
    data = (data - 0.5) / 0.5
    
    # 4. Shaping
    data = np.expand_dims(data, axis=0) # Add channel dim: (1, 512, 768)
    data = np.expand_dims(data, axis=0) # Add batch dim: (1, 1, 512, 768)

    # 3. Run Inference
    output = session.run([output_name], {input_name: data})
    # ... (Post-process 保持不变) ...
    
    # 返回原始 GPU BGR 帧，用于可视化
    return predict, gpu_frame_bgr.download()


if __name__ == '__main__':
    session, input_name, output_name, gpu_clahe_output, gpu_input_float, gpu_resize = init_model()

    while g_running:
        try:
            frame_package = frame_queue.get(timeout=0.1) 
        except queue.Empty:
            continue
        
        raw_data, frame_info = frame_package
        
        # --- 替换 MV_CC_ConvertPixelType ---
        
        # 1. 上传 Raw Data 到 GPU
        # 注意: 假设相机输出的是 Mono8，否则需要自定义 Bayer 转 BGR/Gray 的 CUDA Kernel
        frame_array = np.ctypeslib.as_array(raw_data, shape=(frame_info["height"], frame_info["width"]))
        gpu_raw_frame = cv2.cuda_GpuMat(frame_array)

        # 2. 如果相机输出是 Mono8/Gray，则不需要 cvtColor。
        #    如果相机输出是 Bayer，需要使用 cv2.cuda.cvtColor(..., cv2.COLOR_BAYER_BG2BGR)
        #    这里假设它已经是 Gray（Mono8），可以用于分割预处理。

        # 3. 在 GPU 上执行预处理 (需要一个只接收 GpuMat 的新函数)
        # 暂时跳过，我们使用 run_segmentation 内部处理

        # --- segmentation ---
        # 核心改动：run_segmentation 现在返回预测图和 CPU BGR 帧
        # 我们需要提供 GPU 资源给它
        predict_map, img_bgr = run_segmentation_gpu_optimized(
            gpu_raw_frame, # 使用 GPU Raw 帧作为输入
            session, input_name, output_name, 
            gpu_clahe_output, gpu_input_float, gpu_resize
        )

        # ... (visualization 和 FPS 绘制保持不变) ...
        
        # 原始的 BGR 帧（img_bgr）现在是从 GPU 下载的，用于可视化。
        # 注意：可视化本身会下载数据到 CPU。
        
        # ... (cv2.imshow 和 break 逻辑保持不变) ...

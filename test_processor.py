import numpy as np
import cv2
from eye_guidance_processor import BatchedEyeGuidanceProcessor

print("[Test] 正在初始化 Processor... (如果这是第一次运行，可能需要几分钟)")
try:
    # 确保 redness=False 与你的 .onnx 文件名 (infraredx.onnx) 匹配
    processor = BatchedEyeGuidanceProcessor(redness=False) 
    print("[Test] Processor 初始化完成。")
    
    # 创建一个模拟的黑色帧
    # 尺寸模仿 camera_wrapper.py 的输出 (假设为 1280x1024)
    # 你可以根据你的相机分辨率修改
    frame = np.zeros((512, 768, 3), dtype=np.uint8)
    
    print("[Test] 正在处理一个模拟 batch...")
    # 处理器内部会 @time_it
    display_frames = processor.process_batch(frame, frame.copy())
    
    print("[Test] Batch 处理完成。")
    print(f"[Test] 收到的帧 0 shape: {display_frames[0].shape}")
    print("[Test] 处理器单独测试成功!")

except Exception as e:
    print(f"[Test] 处理器测试失败: {e}")
    import traceback
    traceback.print_exc()
import cv2
import numpy as np
import os


TARGET_W, TARGET_H = 768, 512
IMG_PATH = 'check/widthOrProp_od.jpg'

def preprocessing(gpu_frame_raw):
    # 3. Demosaicing (Bayer RG 8 -> BGR)
    #    !!! 这是一个模拟步骤 !!!
    #    一个真实的 Bayer 帧是单通道的。你需要先上传那个单通道 GpuMat。
    #    例如: 
    #    cpu_frame_raw_bayer = cv2.imread(BAYER_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    #    gpu_frame_gray_raw = cv2.cuda_GpuMat()
    #    gpu_frame_gray_raw.upload(cpu_frame_raw_bayer)
    #
    #    --- 这是你要求演示的代码 (已注释掉) ---
    #    print("Simulating Demosaicing...")
    #    # 假设 gpu_frame_gray_raw 是上传的 Bayer 帧
    #    # gpu_frame_bgr_demosaiced = cv2.cuda.demosaicing(gpu_frame_gray_raw, cv2.COLOR_BayerRG2BGR)

    gpu_frame_bgr_demosaiced = gpu_frame_bgr
    gpu_frame_bgr_resized = cv2.cuda.resize(gpu_frame_bgr_demosaiced, (TARGET_W, TARGET_H))
    gpu_frame_gray = cv2.cuda.cvtColor(gpu_frame_bgr_resized, cv2.COLOR_BGR2GRAY)

    lut_table_cpu = 255.0 * (np.linspace(0, 1, 256)**0.8)
    lut_table_cpu = np.clip(lut_table_cpu, 0, 255).astype(np.uint8)
    gpu_frame_lut = cv2.cuda.LUT(gpu_frame_gray, lut_table_cpu)

    clahe_gpu = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gpu_frame_clahe = clahe_gpu.apply(gpu_frame_lut)

    gpu_frame_normalized = gpu_frame_clahe.convertTo(cv2.CV_32F, alpha=(1/127.5), beta=-1.0)
    gpu_tensor = torch.utils.dlpack.from_dlpack(cv2.cuda.GpuMat.dlpack(gpu_frame_normalized))
    gpu_tensor = gpu_tensor.view(1, 1, TARGET_H, TARGET_W) # (512, 768) -> (1, 1, 512, 768)

    return gpu_frame_bgr_resized, gpu_tensor


def inferencing(input_tensor):
    print(f"NN Input Shape: {input_tensor.shape}") # 应该打印 (1, 1, 512, 768)

    output_tensor = np.random.rand(1, 1, TARGET_H, TARGET_W).astype(np.float32)
    return output_tensor

def postprocessing(gpu_frame_bgr, nn_output):
    return gpu_frame_bgr

def main():

    cpu_frame_raw = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    print(f"Loaded image, shape: {cpu_frame.shape}")
    
    gpu_frame_raw = cv2.cuda_GpuMat()
    gpu_frame_raw.upload(cpu_frame_raw)
    
    gpu_frame_bgr_resized, gpu_tensor = preprocessing(gpu_frame_raw)
    nn_output = inferencing(gpu_tensor)
    gpu_final_image_to_display = post_processing(gpu_frame_bgr_resized, nn_output)

    cpu_final_image_to_display = gpu_final_image_to_display.download()
    cv2.imshow("Final Post-Processed Output", cpu_final_image_to_display)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def __name__ == '__main__':
    main()

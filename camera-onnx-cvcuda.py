import cv2
import numpy as np
import os
import cvcuda


TARGET_W, TARGET_H = 768, 512
IMG_PATH = 'check/widthOrProp_od.jpg'

def preprocessing(cpu_frame_raw):
    istream = cvcuda.Stream()
    gpu_frame_bayer = cvcuda.as_tensor(cpu_frame_raw, "HW")
    gpu_frame_bayer = cvcuda.push_tensor(gpu_frame_bayer, stream=istream)

    gpu_frame_bgr = cvcuda.demosaicing(
        gpu_frame_bayer,
        bayer_pattern=cvcuda.BayerPattern.RG,
        output_format=cvcuda.ColorSpec.BGR8,
        stream=istream
    )

    target_w, target_h = 768, 512
    gpu_frame_bgr_resized = cvcuda.resize(gpu_frame_bgr, (target_w, target_h), interp=cvcuda.Interp.LINEAR, stream=istream)
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
    print(f"Loaded image, shape: {cpu_frame_raw.shape}")
    
    gpu_frame_raw = cv2.cuda_GpuMat()
    gpu_frame_raw.upload(cpu_frame_raw)
    
    gpu_frame_bgr_resized, gpu_tensor = preprocessing(gpu_frame_raw)
    nn_output = inferencing(gpu_tensor)
    gpu_final_image_to_display = postprocessing(gpu_frame_bgr_resized, nn_output)

    cpu_final_image_to_display = gpu_final_image_to_display.download()
    cv2.imshow("Final Post-Processed Output", cpu_final_image_to_display)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def __name__ == '__main__':
    main()

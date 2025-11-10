# -- coding: utf-8 --

import sys
import os
import termios
import queue
import threading
import numpy as np
import cv2
from ctypes import *
import time
import onnxruntime as ort

sys.path.append("./MvImport")
from MvCameraControl_class import *

frame_queue = queue.Queue(maxsize=10)
g_running = True 

# 1. Producer
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = CFUNCTYPE(None, pData, stFrameInfo, c_void_p)

def image_callback(pData, pFrameInfo, pUser):
    global g_running, frame_queue
    if not g_running:
        return

    stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
    if stFrameInfo:
        if pData is None:
            print("Error: pData is None in callback.")
            return
        
        raw_data = (c_ubyte * stFrameInfo.nFrameLen)()
        memmove(byref(raw_data), pData, stFrameInfo.nFrameLen)

        frame_info_copy = {
            "width": stFrameInfo.nWidth,
            "height": stFrameInfo.nHeight,
            "pixel_type": stFrameInfo.enPixelType,
            "frame_len": stFrameInfo.nFrameLen
        }

        try:
            frame_package = (raw_data, frame_info_copy)
            frame_queue.put_nowait(frame_package)
        except queue.Full:
            pass

CALL_BACK_FUN = FrameInfoCallBack(image_callback)

def init_model(redness=False):
    """
    Loads the ONNX model and builds/loads the TensorRT engine.
    This runs only ONCE at startup and may take a few seconds.
    """
    print("Loading model and building/loading TensorRT engine...")
    
    model_path = 'log/weights_infrared_best.onnx'
    if redness:
        model_path = 'log/weights_visible_best.onnx'

    trt_cache_path = os.path.join(os.path.dirname(model_path), "trt_cache")
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


def preprocessing():
    pass


def postprocessing():
    pass


def run_segmentation(frame, session, input_name, output_name):
    # 1. Preprocessing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_resized = cv2.resize(frame_gray, (768, 512)) # W, H

    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    frame_gray_g = cv2.LUT(frame_gray_resized, table)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    frame_gray_c = clahe.apply(np.array(np.uint8(frame_gray_g)))

    # 2. Normalize and shape for the model (H,W) -> (1,1,H,W)
    data = frame_gray_c.astype(np.float32) / 255.0
    data = (data - 0.5) / 0.5
    data = np.expand_dims(data, axis=0) # Add channel dim: (1, 512, 768)
    data = np.expand_dims(data, axis=0) # Add batch dim: (1, 1, 512, 768)

    # 3. Run Inference
    output = session.run([output_name], {input_name: data})
    output_data = output[0] # Shape (1, C, 512, 768)
    
    # 4. Post-process
    predict = np.argmax(output_data, axis=1) # Shape (1, 512, 768)
    predict = np.squeeze(predict) # Shape (512, 768)

    return predict


def visualize_segmentation(predict_map, palette):
    return palette[predict_map].astype(np.uint8)


# --- 2. Consumer ---
if __name__ == "__main__":

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("No devices found!")
        sys.exit()

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[int(0)], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Open device fail!")
        sys.exit()

    print("Loading camera parameters from FeatureFile.ini...")
    ret = cam.MV_CC_FeatureLoad("FeatureFile.ini") #
    if ret != 0:
        print("Failed to load parameters from file! Using current settings.")
    else:
        print("Parameters loaded successfully.")

    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    session, input_name, output_name = init_model()
    palette = np.array([
        [0, 0, 0],    # 类别 0 (背景) - 黑色
        [0, 255, 0],  # 类别 1 - 绿色
        [255, 0, 0],  # 类别 2 - 蓝色
        [0, 0, 255]   # 类别 3 - 红色
    ], dtype=np.uint8)

    ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN, None)
    if ret != 0:
        print("Register image callback fail!")
        sys.exit()

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing fail!")
        sys.exit()

    print("Grabbing... Press 'q' in OpenCV window to stop.")

    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()

    frame_count = 0
    start_time = time.time()
    display_fps = 0
    
    while g_running:
        try:
            frame_package = frame_queue.get(timeout=0.1) 
        except queue.Empty:
            continue
        
        raw_data, frame_info = frame_package
        img_bgr = None
        
        # 处理非BGR8格式
        nRGBSize = frame_info["width"] * frame_info["height"] * 3
        
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = frame_info["width"]
        stConvertParam.nHeight = frame_info["height"]
        stConvertParam.pSrcData = raw_data  # 使用复制好的数据
        stConvertParam.nSrcDataLen = frame_info["frame_len"]
        stConvertParam.enSrcPixelType = frame_info["pixel_type"]
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        
        # 分配目标缓冲区 (优化点：应在循环外完成)
        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nDstBufferSize = nRGBSize

        ret = cam.MV_CC_ConvertPixelType(stConvertParam)
        if ret != 0:
            print(f"Convert pixel type fail! Code: {ret}")
            continue

        img_bgr = np.ctypeslib.as_array(stConvertParam.pDstBuffer, shape=(stConvertParam.nDstLen,))
        img_bgr = img_bgr.reshape((frame_info["height"], frame_info["width"], 3))
        
        # --- segmentation ---
        predict_map = run_segmentation(img_bgr, session, input_name, output_name)

        # --- visualization ---
        img_bgr = cv2.resize(img_bgr, (768, 512))
        seg_map_colored = visualize_segmentation(predict_map, palette)
        final_image = cv2.addWeighted(img_bgr, 0.6, seg_map_colored, 0.4, 0)

        # --- 绘制FPS并显示 ---
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0:
            display_fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        fps_text = f"FPS: {display_fps:.2f}"
        cv2.putText(final_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Realtime Segmentation (Jetson)", final_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            g_running = False
            break

    print("Stopping...")
    cv2.destroyAllWindows()
    ret = cam.MV_CC_StopGrabbing() #
    ret = cam.MV_CC_CloseDevice() #
    ret = cam.MV_CC_DestroyHandle() #
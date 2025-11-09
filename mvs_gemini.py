# -- coding: utf-8 --

import sys
from ctypes import *
import threading
import time

sys.path.append("./MvImport")
from MvCameraControl_class import *

import cv2
import numpy as np

class SimpleMvUSBcamera:
    def __init__(self):
        self.cam = MvCamera()
        self.is_grabbing = False
        self.is_connected = False
        self.nPayloadSize = 0
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.producer_thread = None
        
    def connect_first_usb_device(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_USB_DEVICE
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
        if ret != 0:
            print(f"Error: 枚举设备失败! ret[0x{ret:x}]")
            return False
        
        if device_list.nDeviceNum == 0:
            print("未找到USB设备!")
            return False

        print(f"找到 {device_list.nDeviceNum} 个USB设备. 正在连接第一个...")

        stDeviceList = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print(f"Error: 创建句柄失败! ret[0x{ret:x}]")
            return False

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"Error: 打开设备失败! ret[0x{ret:x}]")
            return False
        
        self.is_connected = True
        
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"Error: 设置触发模式失败! ret[0x{ret:x}]")
            self.close()
            return False

        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print(f"Error: 获取PayloadSize失败! ret[0x{ret:x}]")
            return False
        self.nPayloadSize = stParam.nCurValue
            
        print("相机连接成功.")
        return True
        
    def start_grabbing(self):
        if not self.is_connected:
            print("Error: 相机未连接.")
            return False
        
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"Error: 开始取流失败! ret[0x{ret:x}]")
            return False
            
        self.is_grabbing = True
        
        self.producer_thread = threading.Thread(target=self._producer_loop)
        self.producer_thread.daemon = True
        self.producer_thread.start()
        
        return True

    def _producer_loop(self):
        print("生产者线程已启动...")
        
        data_buf = (c_ubyte * self.nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        while self.is_grabbing:
            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(data_buf), self.nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
                image = self.convert_to_bgr(data_buf, stFrameInfo)
                
                if image is not None:
                    with self.frame_lock:
                        self.latest_frame = image
        
        print("生产者线程已停止.")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def convert_to_bgr(self, data_buf, stFrameInfo):
        pixel_type = PixelType_Gvsp_BGR8_Packed
        nBGRSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
        
        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        
        stConvertParam.nWidth = stFrameInfo.nWidth
        stConvertParam.nHeight = stFrameInfo.nHeight
        stConvertParam.pSrcData = data_buf
        stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
        stConvertParam.enDstPixelType = pixel_type
        stConvertParam.pDstBuffer = (c_ubyte * nBGRSize)()
        stConvertParam.nDstBufferSize = nBGRSize
        
        ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
        if ret != 0:
            print(f"Error: 转换像素格式失败! ret[0x{ret:x}]")
            return None
            
        try:
            img_view = np.ctypeslib.as_array(stConvertParam.pDstBuffer, shape=(nBGRSize,))
            np_image = img_view.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
            return np_image.copy()
            
        except Exception as e:
            print(f"Error: Numpy 转换失败! {e}")
            return None

    def stop_grabbing(self):
        if self.is_grabbing:
            self.is_grabbing = False
            
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=2)
                self.producer_thread = None
        
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print(f"Error: 停止取流失败! ret[0x{ret:x}]")
        else:
            print("SDK已停止取流.")
        
    def close(self):
        self.stop_grabbing()
        
        if self.is_connected:
            self.cam.MV_CC_CloseDevice()
            self.is_connected = False
        
        self.cam.MV_CC_DestroyHandle()
        print("相机资源已释放.")

if __name__ == "__main__":
    
    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print("MVS SDK Version: 0x%x" % SDKVersion)

    cam = SimpleMvUSBcamera()
    
    try:
        if not cam.connect_first_usb_device():
            sys.exit()

        if not cam.start_grabbing():
            sys.exit()
            
        print("\n正在显示图像... 按 'q' 键退出.")
        
        while True:
            image = cam.get_latest_frame()
            
            if image is not None:
                cv2.imshow("Camera Feed", image)
            else:
                time.sleep(0.01)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"发生意外错误: {e}")
    finally:
        print("正在关闭...")
        cv2.destroyAllWindows()
        cam.close()
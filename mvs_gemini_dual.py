# -- coding: utf-8 --

import sys
import os
import threading
import time
from ctypes import *

sys.path.append("./MvImport")
from MvCameraControl_class import *

import cv2
import numpy as np

class CameraWorker:
    def __init__(self, stDevInfo, cpu_id):
        self.cam = MvCamera()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_grabbing = False
        self.is_connected = False
        self.nPayloadSize = 0
        self.producer_thread = None
        self.cpu_id = cpu_id
        self.serial_number = ""

        try:
            # stDevInfo 现在是结构体本身, 不再需要 cast
            self.serial_number = "".join([chr(per) for per in stDevInfo.SpecialInfo.stUsb3VInfo.chSerialNumber if per != 0])
        except:
            self.serial_number = "Unknown_Cam"

        # 这里传递结构体, ctypes 会自动处理 byref
        ret = self.cam.MV_CC_CreateHandle(stDevInfo)
        if ret != 0:
            raise RuntimeError(f"创建句柄失败 (Cam {self.serial_number}): ret[0x{ret:x}]")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"打开设备失败 (Cam {self.serial_number}): ret[0x{ret:x}]")

        self.is_connected = True

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"Warning: 设置触发模式失败 (Cam {self.serial_number})")

        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise RuntimeError(f"获取PayloadSize失败 (Cam {self.serial_number}): ret[0x{ret:x}]")
        
        self.nPayloadSize = stParam.nCurValue
        print(f"相机 {self.serial_number} 连接成功, 将分配到 CPU {self.cpu_id}")

    def start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始取流失败 (Cam {self.serial_number}): ret[0x{ret:x}]")

        self.is_grabbing = True
        self.producer_thread = threading.Thread(target=self._producer_loop)
        self.producer_thread.daemon = True
        self.producer_thread.start()

    def _producer_loop(self):
        try:
            os.sched_setaffinity(0, {self.cpu_id})
            print(f"生产者线程 (Cam {self.serial_number}) 已成功绑定到 CPU {self.cpu_id}")
        except Exception as e:
            print(f"Warning: 绑定 CPU {self.cpu_id} 失败 (Cam {self.serial_number}): {e}. 仅在 Linux 上支持.")

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
        
        print(f"生产者线程 (Cam {self.serial_number}) 已停止.")

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
            return None

        try:
            img_view = np.ctypeslib.as_array(stConvertParam.pDstBuffer, shape=(nBGRSize,))
            np_image = img_view.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
            return np_image.copy()
        except Exception:
            return None

    def stop(self):
        if self.is_grabbing:
            self.is_grabbing = False
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=2)
                self.producer_thread = None

        if self.is_connected:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_connected = False
        
        print(f"相机 {self.serial_number} 资源已释放.")

class MultiCameraSystem:
    def __init__(self):
        self.workers = []
        try:
            self.cpu_count = len(os.sched_getaffinity(0))
        except:
            print("Warning: 无法获取 CPU 亲和性, 回退到 os.cpu_count()")
            self.cpu_count = os.cpu_count() or 1
        
        print(f"系统检测到 {self.cpu_count} 个可用 CPU 核心.")

    def discover_and_connect(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_USB_DEVICE
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
        if ret != 0:
            raise RuntimeError(f"枚举设备失败! ret[0x{ret:x}]")

        if device_list.nDeviceNum == 0:
            print("未找到USB设备!")
            return

        print(f"找到 {device_list.nDeviceNum} 个USB设备.")

        for i in range(device_list.nDeviceNum):
            # *** 关键修复 ***
            # 我们必须传递结构体 (.contents), 而不是指针
            stDevInfo = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            # (i + 1) % self.cpu_count 确保我们跳过 CPU 0 (通常用于系统)
            # 并在核心之间循环分配
            cpu_id = (i + 1) % self.cpu_count 
            
            try:
                # 将结构体传递给构造函数
                worker = CameraWorker(stDevInfo, cpu_id)
                self.workers.append(worker)
            except Exception as e:
                print(f"连接相机 {i} 失败: {e}")

    def start_all(self):
        for worker in self.workers:
            worker.start()

    def get_all_frames(self):
        frames = []
        for worker in self.workers:
            frames.append(worker.get_latest_frame())
        return frames

    def stop_all(self):
        for worker in self.workers:
            worker.stop()
        print("所有相机已停止.")

def main():
    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print("MVS SDK Version: 0x%x" % SDKVersion)
    
    system = MultiCameraSystem()
    
    try:
        system.discover_and_connect()
        
        if not system.workers:
            print("没有相机连接成功, 退出.")
            return

        system.start_all()
            
        print("\n正在显示图像... 按 'q' 键退出.")
        
        while True:
            frames = system.get_all_frames()
            valid_frames = [f for f in frames if f is not None]

            if not valid_frames:
                time.sleep(0.01)
                continue
            
            display_img = None
            if len(valid_frames) == 1:
                display_img = valid_frames[0]
            else:
                try:
                    min_h = min(f.shape[0] for f in valid_frames)
                    resized_frames = [cv2.resize(f, (int(f.shape[1] * min_h / f.shape[0]), min_h)) for f in valid_frames]
                    display_img = np.hstack(resized_frames)
                except Exception as e:
                    print(f"图像拼接失败: {e}")
                    time.sleep(0.01)
                    continue

            if display_img is not None:
                cv2.imshow("Multi-Camera Feed", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"发生意外错误: {e}")
    finally:
        print("正在关闭...")
        cv2.destroyAllWindows()
        system.stop_all()

if __name__ == "__main__":
    main()
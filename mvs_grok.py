# -- coding: utf-8 --
import sys
import cv2
import numpy as np
from ctypes import *
from typing import Optional

sys.path.append("./MvImport")
from MvCameraControl_class import *


class UsbCameraLive:
    """
    简洁的 USB 相机实时显示类
    - 只连接第一个 USB 相机
    - 不保存任何文件
    - 直接把 Bayer/RAW 转 RGB 并用 OpenCV 显示
    """

    def __init__(self):
        self.cam: Optional[MvCamera] = None
        self.payload_size: int = 0
        self.width: int = 0
        self.height: int = 0

    def __enter__(self):
        self._open_first_usb_device()
        self._configure()
        self._start_grabbing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------
    # 内部实现
    # -------------------------------------------------
    def _open_first_usb_device(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            raise RuntimeError(f"未发现 USB 相机，ret=0x{ret:x}")

        # 取第一个 USB 设备
        stDeviceInfo = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(stDeviceInfo)
        if ret != 0:
            raise RuntimeError(f"创建句柄失败，ret=0x{ret:x}")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"打开设备失败，ret=0x{ret:x}")

    def _configure(self):
        # 关闭触发
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise RuntimeError(f"设置触发模式失败，ret=0x{ret:x}")

        # 获取 PayloadSize
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise RuntimeError(f"获取 PayloadSize 失败，ret=0x{ret:x}")
        self.payload_size = stParam.nCurValue

    def _start_grabbing(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始取流失败，ret=0x{ret:x}")

    # -------------------------------------------------
    # 公共接口
    # -------------------------------------------------
    def grab_and_show(self) -> bool:
        """
        抓取一帧并实时显示
        返回 False 表示用户按 'q' 或 ESC 退出
        """
        if not self.cam:
            return False

        # 1. 取帧
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        data_buf = (c_ubyte * self.payload_size)()

        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(data_buf), self.payload_size,
                                                stFrameInfo, 1000)
        if ret != 0:
            print(f"Warning: 取帧超时，ret=0x{ret:x}")
            return True

        self.width, self.height = stFrameInfo.nWidth, stFrameInfo.nHeight

        # 2. 像素格式转换 → RGB8
        rgb_size = self.width * self.height * 3
        stConvert = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(stConvert), 0, sizeof(stConvert))
        stConvert.nWidth = self.width
        stConvert.nHeight = self.height
        stConvert.pSrcData = data_buf
        stConvert.nSrcDataLen = stFrameInfo.nFrameLen
        stConvert.enSrcPixelType = stFrameInfo.enPixelType
        stConvert.enDstPixelType = PixelType_Gvsp_RGB8_Packed
        stConvert.pDstBuffer = (c_ubyte * rgb_size)()
        stConvert.nDstBufferSize = rgb_size

        ret = self.cam.MV_CC_ConvertPixelType(stConvert)
        if ret != 0:
            print(f"Warning: 像素转换失败，ret=0x{ret:x}")
            return True

        # 3. 转 numpy + BGR (OpenCV 默认 BGR)
        rgb_buf = bytes(stConvert.pDstBuffer[:stConvert.nDstLen])
        img = np.frombuffer(rgb_buf, dtype=np.uint8).reshape(self.height, self.width, 3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 4. 显示
        cv2.imshow("USB Camera Live (press 'q' or ESC to quit)", bgr)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), 27)   # 27 = ESC

    def close(self):
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"SDK Version: 0x{MvCamera.MV_CC_GetSDKVersion():x}")

    with UsbCameraLive() as live:
        print("开始实时显示，按 'q' 或 ESC 退出...")
        while live.grab_and_show():
            pass

    print("已退出。")
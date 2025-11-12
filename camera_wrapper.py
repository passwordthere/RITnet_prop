import sys
import numpy as np
from ctypes import *

sys.path.append("./MvImport")
from MvCameraControl_class import *

class CameraWrapper:
    """
    一个高性能的 MvCamera 包装器。
    在 __init__ 中处理连接和开始抓取。
    get_latest_frame() 获取并转换最新的帧为 numpy 数组。
    """
    def __init__(self, camera_index=0):
        print(f"[CAM {camera_index}] 正在初始化...")
        self.cam = MvCamera()
        self.camera_index = camera_index
        self.data_buf = None
        self.nPayloadSize = 0
        self.closed = False

        # 转换参数 (在第一次 get_frame 时初始化)
        self.stConvertParam = None
        self.rgb_buffer = None
        self.width = 0
        self.height = 0

        # 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise Exception(f"[CAM {camera_index}] 枚举设备失败! ret[0x{ret:x}]")

        if deviceList.nDeviceNum == 0:
            raise Exception(f"[CAM {camera_index}] 未找到设备!")
        
        if camera_index >= deviceList.nDeviceNum:
            raise Exception(f"[CAM {camera_index}] 索引 {camera_index} 超出范围 (找到 {deviceList.nDeviceNum} 个设备)")

        print(f"[CAM {camera_index}] 找到 {deviceList.nDeviceNum} 个设备. 正在连接索引 {camera_index}...")

        # 选择设备并创建句柄
        stDeviceList = cast(deviceList.pDeviceInfo[camera_index], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise Exception(f"[CAM {camera_index}] 创建句柄失败! ret[0x{ret:x}]")

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"[CAM {camera_index}] 打开设备失败! ret[0x{ret:x}]")

        # GigE 相机特定设置
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print(f"[CAM {camera_index}] 警告: 设置包大小失败! ret[0x{ret:x}]")
            else:
                print(f"[CAM {camera_index}] 警告: 获取包大小失败! ret[0x{nPacketSize:x}]")

        # 设置触发模式为 off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise Exception(f"[CAM {camera_index}] 设置触发模式失败! ret[0x{ret:x}]")

        # 获取 PayloadSize
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise Exception(f"[CAM {camera_index}] 获取 PayloadSize 失败! ret[0x{ret:x}]")
        
        self.nPayloadSize = stParam.nCurValue
        self.data_buf = (c_ubyte * self.nPayloadSize)() # 预分配原始数据缓冲区

        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"[CAM {camera_index}] 开始取流失败! ret[0x{ret:x}]")
            
        print(f"[CAM {camera_index}] 初始化完成并开始取流.")

    def get_latest_frame(self):
        """
        抓取一帧, 将其转换为 RGB, 并作为 numpy 数组返回。
        """
        if self.closed:
            return None

        stDeviceList = MV_FRAME_OUT_INFO_EX()
        memset(byref(stDeviceList), 0, sizeof(stDeviceList))

        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.nPayloadSize, stDeviceList, 1000)
        
        if ret == 0:
            # --- 第一次运行时初始化转换参数 ---
            if self.rgb_buffer is None:
                print(f"[CAM {self.camera_index}] 第一次抓取: Width[{stDeviceList.nWidth}], Height[{stDeviceList.nHeight}]")
                self.width = stDeviceList.nWidth
                self.height = stDeviceList.nHeight
                nRGBSize = self.width * self.height * 3
                
                # 预分配 RGB 缓冲区
                self.rgb_buffer = (c_ubyte * nRGBSize)()
                
                # 设置固定的转换参数
                self.stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(self.stConvertParam), 0, sizeof(self.stConvertParam))
                self.stConvertParam.nWidth = self.width
                self.stConvertParam.nHeight = self.height
                self.stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                self.stConvertParam.pDstBuffer = self.rgb_buffer
                self.stConvertParam.nDstBufferSize = nRGBSize
            
            # --- 更新每帧变化的转换参数 ---
            self.stConvertParam.pSrcData = self.data_buf
            self.stConvertParam.nSrcDataLen = stDeviceList.nFrameLen
            self.stConvertParam.enSrcPixelType = stDeviceList.enPixelType
            
            # 转换
            ret = self.cam.MV_CC_ConvertPixelType(self.stConvertParam)
            if ret != 0:
                print(f"[CAM {self.camera_index}] 转换像素失败! ret[0x{ret:x}]")
                return None
            
            # --- 转换为 NumPy 数组 ---
            # 从 C 缓冲区创建 NumPy 数组 (无拷贝)
            img_buff = (c_ubyte * self.stConvertParam.nDstLen).from_buffer(self.rgb_buffer)
            image = np.frombuffer(img_buff, dtype=np.uint8)
            # Reshape (无拷贝)
            image = image.reshape((self.height, self.width, 3))
            
            # 返回一个拷贝, 这样内部缓冲区就可以被下一次抓取安全地覆盖
            return image.copy()
            
        else:
            print(f"[CAM {self.camera_index}] 获取帧失败! ret[0x{ret:x}]")
            return None

    def close(self):
        """
        停止抓取, 关闭并销毁句柄。
        """
        if not self.closed:
            print(f"[CAM {self.camera_index}] 正在关闭...")
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.closed = True
            print(f"[CAM {self.camera_index}] 已关闭.")

    def __del__(self):
        # 确保在对象销毁时资源被释放
        self.close()
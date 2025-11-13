import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import sys
import signal
from camera_wrapper import CameraWrapper

# -------------------------------------------------------------------
# 核心工作函数 (必须是顶层函数，以便 multiprocessing 'pickle' 它)
# -------------------------------------------------------------------

def _camera_worker(camera_index, shm_name, shape, dtype, stop_event):
    """
    (内部函数) 在子进程中运行。
    实例化 CameraWrapper, 循环抓取帧, 并将它们写入共享内存。
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 忽略 Ctrl+C, 由主进程处理
    
    shm = None
    cam = None
    try:
        # 附加到现有的共享内存
        shm = shared_memory.SharedMemory(name=shm_name)
        # 创建一个 NumPy 数组 "视图", 直接映射到共享内存
        shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # 初始化相机 (在子进程中)
        cam = CameraWrapper(camera_index)

        print(f"[Worker {camera_index}] 开始循环抓取...")
        while not stop_event.is_set():
            frame = cam.get_latest_frame()
            
            if frame is not None:
                # 将帧数据复制到共享内存
                np.copyto(shm_array, frame)
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"[Worker {camera_index}] 发生错误: {e}")
    finally:
        # 清理
        if cam:
            cam.close()
        if shm:
            shm.close()
        print(f"[Worker {camera_index}] 退出.")

# -------------------------------------------------------------------
# 管理器类 (封装所有复杂性)
# -------------------------------------------------------------------

class MultiCameraManager:
    """
    一个高级管理器, 用于处理多个 CameraWrapper 实例在
    独立的子进程中, 并通过共享内存返回它们的帧。

    使用 'with' 语句可确保自动清理：
    
    with MultiCameraManager(indices=[0, 1]) as manager:
        while True:
            frames = manager.get_frames()
            if frames:
                # frame_0 = frames[0]
                # frame_1 = frames[1]
                ...
    """
    def __init__(self, camera_indices=[0, 1]):
        print(f"[Manager] 正在初始化 {len(camera_indices)} 个相机...")
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        
        self.camera_indices = camera_indices
        self.processes = []
        self.shm_list = []
        self.shm_views = {}
        self.stop_event = mp.Event()
        
        try:
            # --- 1. 获取信息并创建共享内存 (原 get_camera_info + main 逻辑) ---
            for idx in self.camera_indices:
                info = self._get_camera_info(idx)
                if info is None:
                    raise Exception(f"无法启动相机 {idx}, 退出。")
                
                shape = info['shape']
                dtype = info['dtype']
                nbytes = np.prod(shape) * np.dtype(dtype).itemsize

                # 创建共享内存块
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self.shm_list.append(shm)
                
                # 为主进程创建此共享内存的 numpy 视图
                self.shm_views[idx] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                
                # 准备启动工作进程 (原 camera_worker)
                p = mp.Process(target=_camera_worker, 
                               args=(idx, shm.name, shape, dtype, self.stop_event))
                self.processes.append(p)

            # --- 2. 启动所有工作进程 (原 main 逻辑) ---
            print("[Manager] 正在启动所有工作进程...")
            for p in self.processes:
                p.start()
            
            print(f"[Manager] 初始化成功。{len(self.shm_views)} 个相机流已激活。")

        except Exception as e:
            print(f"[Manager] 初始化失败: {e}")
            self.close() # 如果启动失败，立即清理
            raise  # 重新抛出异常，以便调用者知道失败了

    def _get_camera_info(self, camera_index):
        """
        (私有方法) 连接相机一次以获取其尺寸。
        这在主进程中运行。
        """
        print(f"[Manager] 正在获取相机 {camera_index} 的信息...")
        cam = None
        try:
            cam = CameraWrapper(camera_index)
            frame = cam.get_latest_frame()
            if frame is None:
                raise Exception(f"无法从相机 {camera_index} 获取帧")
            
            shape = frame.shape
            dtype = frame.dtype
            print(f"[Manager] 相机 {camera_index} 信息: {shape}, {dtype}")
            return {'shape': shape, 'dtype': dtype}
        except Exception as e:
            print(f"[Manager] 获取相机 {camera_index} 信息时出错: {e}")
            return None
        finally:
            if cam:
                cam.close()
                del cam

    def get_frames(self):
        """
        以零拷贝方式返回所有相机最新帧的字典。
        
        返回:
            dict: {camera_index: numpy_array_view, ...}
                  例如: {0: frame_0, 1: frame_1}
        """
        # 这只是一个属性访问, 几乎是瞬时的
        return self.shm_views

    def close(self):
        """
        停止所有子进程并清理共享内存。
        """
        print("[Manager] 正在关闭...")
        if self.stop_event:
            self.stop_event.set() # 通知所有 worker 停止
        
        for p in self.processes:
            p.join(timeout=5) # 等待进程退出
            if p.is_alive():
                print(f"[Manager] 进程 {p.pid} 未能正常退出, 正在终止。")
                p.terminate()

        print("[Manager] 正在清理共享内存...")
        for shm in self.shm_list:
            shm.close()     # 关闭主进程的句柄
            shm.unlink()    # 销毁共享内存块
        
        self.shm_list = []
        self.processes = []
        self.shm_views = {}
        print("[Manager] 清理完成。")

    # --- 添加 'with' 语句支持 ---

    def __enter__(self):
        """ 允许 'with' 语句： `with MultiCameraManager() as m:` """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ 'with' 语句退出时自动调用 close() """
        self.close()


if __name__ == "__main__":
    print("主程序启动。正在初始化 MultiCameraManager...")
    print("按 Ctrl+C 停止。")

    # 使用 'with' 语句来自动管理启动和关闭
    # 这会调用 __init__ 并启动所有进程
    try:
        with MultiCameraManager(camera_indices=[0, 1]) as manager:
            
            print("\n--- 管理器已启动并运行 ---")
            
            # 主循环 (例如你的推理循环)
            while True:
                # 1. 以零延迟获取所有帧
                frames = manager.get_frames()
                
                # 2. 按索引访问它们
                frame_0 = frames[0]  # 来自相机 0 的视图
                frame_1 = frames[1]  # 来自相机 1 的视图
                
                # 警告：frame_0 和 frame_1 是共享内存的*视图*。
                # 它们可能在你处理它们时被子进程覆盖。
                # 进行推理前，最好先做一个拷贝：
                # frame_0_copy = frame_0.copy()
                # frame_1_copy = frame_1.copy()
                
                # 示例处理：
                print(f"Cam 0 Mean: {np.mean(frame_0):.2f} | Cam 1 Mean: {np.mean(frame_1):.2f}   ", end="\r")

                # 在这里运行你的推理：
                # result = model(frame_0_copy, frame_1_copy)
                
                # 模拟推理/显示循环
                time.sleep(0.01) # 模拟高速循环

    except KeyboardInterrupt:
        print("\n[Main] 收到 Ctrl+C, 正在退出。 'with' 语句将自动调用 close().")
    except Exception as e:
        print(f"\n[Main] 发生未处理的错误: {e}")
    
    print("[Main] 程序已干净地退出。")
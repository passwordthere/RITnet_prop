import cv2
import numpy as np
import os
import math
import onnxruntime as ort
import time
import sys
from functools import wraps

from camera_manager import MultiCameraManager


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"Time for {func.__name__}: {(t2 - t1) * 1000:.2f} ms")
        return result
    return wrapper

class BatchedEyeGuidanceProcessor:
    def __init__(self, redness=False, offset_x=20, offset_y=20):
        print("[Processor] EyeGuidanceProcessor 正在初始化...")

        self.session, self.input_name, self.output_name = self._init_model(redness)

        self._target_size = (768, 512)
        self._lut_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype('uint8')
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

        self.frame_id = 0
        self.offset_x = offset_x
        self.offset_y = offset_y

        print("[Processor] 初始化完成。")

    def _init_model(self, redness=False):
        model_path = 'infraredx.onnx'
        if redness:
            model_path = 'visible.onnx'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        trt_cache_path = os.path.join(os.path.dirname(model_path), "trt_cache")
        os.makedirs(trt_cache_path, exist_ok=True)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

        try:
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            # session = ort.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            print(f"[Processor] 模型加载! Provider: {session.get_providers()}")
            return session, input_name, output_name
        except Exception as e:
            print(f"[Processor] 加载模型失败: {e}")
            raise

    def process_batch(self, frame_0: np.ndarray, frame_1: np.ndarray) -> list:
        input_batch, resized_frames = self._preprocess_batch(frame_0, frame_1)

        raw_output_batch = self._inference(input_batch)

        display_frames = self._postprocess_batch(raw_output_batch, resized_frames)

        self.frame_id += 1

        return display_frames

    def _preprocess_batch(self, frame_0, frame_1):
        p_frame_0, r_frame_0 = self._preprocess_single(frame_0)
        p_frame_1, r_frame_1 = self._preprocess_single(frame_1)

        input_batch = np.concatenate((p_frame_0, p_frame_1), axis=0)

        resized_frames = [r_frame_0, r_frame_1]

        return input_batch, resized_frames

    def _postprocess_batch(self, raw_output_batch, resized_frames):
        display_frame_0 = self._postprocess_single(
            raw_output_batch[0:1],
            resized_frames[0]
        )

        display_frame_1 = self._postprocess_single(
            raw_output_batch[1:2],
            resized_frames[1]
        )

        return [display_frame_0, display_frame_1]

    def _preprocess_single(self, frame: np.ndarray) -> (np.ndarray, np.ndarray):
        frame_resized = cv2.resize(frame, self._target_size)

        if frame_resized.ndim == 3 and frame_resized.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = cv2.resize(frame, self._target_size)
            frame_resized = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        frame_uint8 = cv2.LUT(frame_gray, self._lut_table)
        frame_clahe = self._clahe.apply(frame_uint8)

        data = frame_clahe.astype(np.float32) / 255.0
        data = (data - 0.5) / 0.5
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)

        return data, frame_resized

    @time_it
    def _inference(self, input_data: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]

    def _postprocess_single(self, raw_output: np.ndarray, original_resized_frame: np.ndarray) -> np.ndarray:
        predict = np.argmax(raw_output, axis=1)
        pred_cpu = np.squeeze(predict)

        display_frame = self._draw_target_crosshair(original_resized_frame, pred_cpu)

        return display_frame

    @staticmethod
    def _get_outermost_sclera_point(predict: np.ndarray) -> (tuple or None):
        y_coords, x_coords = np.where(predict == 1)
        if len(x_coords) == 0:
            return None

        outer_corner_x = np.min(x_coords)
        outer_corner_y = y_coords[x_coords == outer_corner_x]
        outer_corner_y = int(np.mean(outer_corner_y))
        return (outer_corner_x, outer_corner_y)

    def _draw_target_crosshair(self, frame: np.ndarray, predict: np.ndarray) -> np.ndarray:

        frame_with_guide = frame.copy()
        sclera_point = self._get_outermost_sclera_point(predict)

        if sclera_point is None:
            return frame_with_guide

        sclera_x, sclera_y = sclera_point
        target_x = sclera_x - self.offset_x
        target_y = sclera_y + self.offset_y

        pulse_factor = (math.sin(self.frame_id * 0.1) + 1) / 2
        base_color_bgr = (255, 200, 0)

        brightness_min = 0.5
        dynamic_alpha = brightness_min + (1 - brightness_min) * pulse_factor

        pulsing_color = (
            int(base_color_bgr[0] * dynamic_alpha),
            int(base_color_bgr[1] * dynamic_alpha),
            int(base_color_bgr[2] * dynamic_alpha)
        )

        thickness = 1
        radius = 8
        cv2.circle(frame_with_guide, (target_x, target_y), radius, pulsing_color, thickness)
        size = 12
        gap = 4

        cv2.line(frame_with_guide, (target_x - gap, target_y - gap), (target_x - gap - size, target_y - gap), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x - gap, target_y - gap), (target_x - gap, target_y - gap - size), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x + gap, target_y - gap), (target_x + gap + size, target_y - gap), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x + gap, target_y - gap), (target_x + gap, target_y - gap - size), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x - gap, target_y + gap), (target_x - gap - size, target_y + gap), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x - gap, target_y + gap), (target_x - gap, target_y + gap + size), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x + gap, target_y + gap), (target_x + gap + size, target_y + gap), pulsing_color, thickness)
        cv2.line(frame_with_guide, (target_x + gap, target_y + gap), (target_x + gap, target_y + gap + size), pulsing_color, thickness)

        return frame_with_guide

if __name__ == "__main__":

    print("[Main] 启动双相机推理流水线...")
    print("[Main] 按 'q' 键退出。")

    try:
        processor = BatchedEyeGuidanceProcessor(redness=False)

        with MultiCameraManager(camera_indices=[0, 1]) as manager:

            print("\n[Main] 管理器已启动。进入主循环...")

            while True:
                frames = manager.get_frames()

                frame_0 = frames[0].copy()
                frame_1 = frames[1].copy()

                display_frames = processor.process_batch(frame_0, frame_1)

                combined_display = np.hstack((display_frames[0], display_frames[1]))

                cv2.imshow("real", combined_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[Main] 收到 Ctrl+C (KeyboardInterrupt), 正在关闭...")
    except FileNotFoundError as e:
        print(f"\n[Main] 错误: {e}")
        print("请确保 .onnx 模型文件在正确的路径下。")
    except Exception as e:
        print(f"\n[Main] 发生未处理的错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("[Main] 程序已干净地退出。")
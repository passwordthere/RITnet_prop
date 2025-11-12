import cv2
import numpy as np
import os
import math
import onnxruntime as ort
import time
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"Time for {func.__name__}: {(t2 - t1) * 1000:.2f} ms")
        return result
    return wrapper

class EyeGuidanceProcessor:
    def __init__(self, redness=False, offset_x=20, offset_y=20):
        print("EyeGuidanceProcessor initializing...")
        
        self.session, self.input_name, self.output_name = self._init_model(redness)
        
        self._target_size = (768, 512)
        self._lut_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype('uint8')
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        
        self.frame_id = 0
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        print("Initialization complete.")

    def _init_model(self, redness=False):
        model_path = 'infrared_fp16.onnx'
        if redness:
            model_path = 'visible.onnx'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

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
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            print(f"Model loaded! Provider: {session.get_providers()}")
            return session, input_name, output_name
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Ensure ONNX Runtime TensorRT dependencies are correctly installed and the model file is valid.")
            raise

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        
        input_data, resized_frame = self._preprocess(frame)
        
        raw_output = self._inference(input_data)
        
        display_frame = self._postprocess(raw_output, resized_frame)
        
        self.frame_id += 1
        
        return display_frame

    def _preprocess(self, frame: np.ndarray) -> (np.ndarray, np.ndarray):
        
        frame_resized = cv2.resize(frame, self._target_size)

        if frame_resized.ndim == 3 and frame_resized.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = cv2.resize(frame, self._target_size)
            frame_resized = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            
        frame_uint8 = cv2.LUT(frame_gray, self._lut_table)
        frame_clahe = self._clahe.apply(frame_uint8)
        
        data = frame_clahe.astype(np.float16) / 255.0
        data = (data - 0.5) / 0.5
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)
        
        return data, frame_resized

    @time_it
    def _inference(self, input_data: np.ndarray) -> np.ndarray:
        
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]

    def _postprocess(self, raw_output: np.ndarray, original_resized_frame: np.ndarray) -> np.ndarray:
        
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

    @time_it
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
    
    processor = EyeGuidanceProcessor(redness=False)

    IMAGE_DIR = 'mock_camera_od/images/'
    image_files = sorted([
        os.path.join(IMAGE_DIR, f) 
        for f in os.listdir(IMAGE_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    frame_id = 0
    print("按 'q' 键退出...")
    while True:
        img_path = image_files[frame_id % len(image_files)]
        frame = cv2.imread(img_path)
        final_frame = processor.process_frame(frame)
        frame_id += 1
        cv2.imshow("Processed Frame", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
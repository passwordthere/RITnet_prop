import os
import cv2
import numpy as np
from camera_manager import MultiCameraManager
from eye_guidance_processor import BatchedEyeGuidanceProcessor


def main():
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

                cv2.imshow("双相机实时推理 (按 'q' 退出)", combined_display)

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


def main_mock_frame():
    processor = BatchedEyeGuidanceProcessor(redness=False)

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
        frame_0 = cv2.imread(img_path)
        frame_1 = frame_0.copy()
        display_frames = processor.process_batch(frame_0, frame_1)
        combined_display = np.hstack((display_frames[0], display_frames[1]))
        frame_id += 1
        cv2.imshow("real", combined_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    main()  # 用两个相机
    # main_mock_frame()   # 如果没有相机，使用离线数据
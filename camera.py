import os
import sys
import cv2
import numpy as np
from MvImport.MvCameraControl_class import MvCamera


def enumerate_devices(cam):
    device_list = cam.EnumDevices()
    if device_list.nDeviceNum == 0:
        print("No camera devices found.")
        sys.exit()
    return device_list


def is_device_accessible(cam, device_info):
    return cam.IsDeviceAccessible(device_info, 1)


def create_device_handle(cam, device_info):
    return cam.CreateHandle(device_info)


def open_device(cam):
    return cam.Open()


def set_get_parameters(cam):
    exposure_value = {'ExposureTime': 5000.0}
    cam.SetFloatValue("ExposureTime", exposure_value["ExposureTime"])
    val = cam.GetFloatValue("ExposureTime")
    print("ExposureTime set and confirmed:", val)


def start_grabbing(cam):
    cam.StartGrabbing()
    print("Camera started grabbing...")


# Frame 中转站
def set_image_node_num(cam, num=10):
    cam.SetImageNodeNum(num)
    print(f"Image node buffer count set to {num}")


def display_camera_feed(cam):
    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame_data, frame_info = cam.GetOneFrameTimeout(1024 * 1024, 1000)
            if ret == 0:
                img = np.frombuffer(frame_data, dtype=np.uint8)
                img = img.reshape((frame_info.nHeight, frame_info.nWidth, 1))  # Mono8
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Live Camera Feed", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received.")
    except Exception as e:
        print("Error during feed display:", e)


def stop_grabbing(cam):
    cam.StopGrabbing()
    print("Stopped grabbing.")


def close_device(cam):
    cam.Close()
    print("Device closed.")


def destroy_handle(cam):
    cam.DestroyHandle()
    print("Handle destroyed.")


def main():
    cam = MvCamera()
    device_list = enumerate_devices(cam)
    for i in range(device_list.nDeviceNum):
        device_info = device_list.pDeviceInfo[i]
        accessible = is_device_accessible(cam, device_info)
        if accessible == 1:
            print(f"Device [{i}] is accessible.")
        else:
            print(f"Device [{i}] is not accessible.")
            continue

    selected_device = device_list.pDeviceInfo[0]
    create_device_handle(cam, selected_device)
    open_device(cam)
    set_image_node_num(cam, 10)  # Optional buffer config
    set_get_parameters(cam)      # Optional parameter config
    start_grabbing(cam)
    display_camera_feed(cam)
    stop_grabbing(cam)
    close_device(cam)
    destroy_handle(cam)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
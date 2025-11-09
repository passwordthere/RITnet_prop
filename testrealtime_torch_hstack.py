import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import model_dict


class CustomPreprocess:
    def __init__(self, target_size=(768, 512)):
        self.target_size = target_size
        self.table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        frame = cv2.resize(frame, self.target_size)
        frame_uint8 = cv2.LUT(frame, self.table.astype('uint8'))    # 注意：LUT 操作需要 uint8 输入
        frame_clahe = self.clahe.apply(frame_uint8)                 # clahe.apply 也需要 uint8 输入
        
        return frame_clahe

inference_transform = transforms.Compose([
    CustomPreprocess(target_size=(768, 512)), # 应用自定义的 cv2 预处理
    transforms.ToTensor(), # (H, W) np.uint8 -> (1, H, W) torch.float32
    transforms.Normalize([0.5], [0.5]) # 标准化
])

def get_predictions(output):
    bs,c,h,w = output.size()
    indices = torch.argmax(output, dim=1) # (B, H, W)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices

def run_realtime_inference(model, device, transform):
    IMAGE_DIR = 'mock_camera_od/images/'
    image_files = sorted([
            os.path.join(IMAGE_DIR, f) 
            for f in os.listdir(IMAGE_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    print("开始实时推理... 按 'q' 退出。")
    
    with torch.no_grad():
        for img_path in image_files:
            frame = cv2.imread(img_path)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_tensor = transform(gray_frame)
            data = image_tensor.unsqueeze(0).to(device) # shape: [1, 1, 512, 768]
            output = model(data) # shape: [1, NumClasses, 512, 768]
            predict = get_predictions(output) 

            inp_gpu = data[0].squeeze() * 0.5 + 0.5 # [512, 768]
            img_orig_gpu = torch.clamp(inp_gpu, 0, 1)
            pred_gpu = predict[0].squeeze() / 3.0 # [512, 768], [0, 1] 范围
            combine_gpu = torch.hstack([img_orig_gpu, pred_gpu])
            display_frame = (combine_gpu * 255.0).byte().cpu().numpy()
            cv2.imshow('real', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


# @profile
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_dict['densenet']
    model = model.to(device)
    filename = 'infrared.pkl'
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    print(f"模型的默认数据类型是: {next(model.parameters()).dtype}")

    # 1. 运行离线数据集
    # print("运行离线数据集...")
    # run_offline_inference(model, device) # 你需要把你的 main() 循环放进这个函数

    # 2. 运行实时摄像头
    print("运行实时推理...")
    run_realtime_inference(model, device, inference_transform)


if __name__ == '__main__':
    
    main()
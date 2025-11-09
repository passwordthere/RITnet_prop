#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""
import torch
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
import os
from idataset import InferenceDataset
from models import model_dict
# from utils import get_predictions
import cv2
import kornia.morphology as morph
# from line_profiler import LineProfiler 


IMAGE_DIR = 'dataset/test/images' 
BATCH_SIZE = 4
NUM_WORKERS = 4

def get_predictions(output):
    values, indices = output.max(1) 
    bs,c,h,w = output.size()
    indices = indices.view(bs,h,w)
    return indices


# @profile
def main():    
    device=torch.device("cuda")
    model = model_dict['densenet']
    model  = model.to(device)
    filename = 'infrared.pkl'
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    print(f"模型的默认数据类型是: {next(model.parameters()).dtype}")

    test_set = InferenceDataset(root_dir=IMAGE_DIR)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    counter=0
    
    os.makedirs('test',exist_ok=True)
    
    kernel_gpu = torch.ones(3, 3).to(device)
    with torch.no_grad():
        for i, batchdata in enumerate(testloader):
            img, img_path = batchdata
            data = img.to(device)       
            output = model(data)            
            predict = get_predictions(output)
            
            for j in range(len(batchdata[0])):  
                predict_gpu = predict[j]
                non_zero_mask = (predict_gpu != 0)
                binary_mask_gpu = (non_zero_mask == 1).float()
                kornia_input = binary_mask_gpu.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
                outline_mask_gpu = morph.gradient(kornia_input, kernel_gpu)
                outline_mask_bool_gpu = outline_mask_gpu.squeeze().bool()

                img_gpu = data[j].squeeze()
                img_gpu_255 = torch.clamp((img_gpu * 0.5 + 0.5) * 255, 0, 255)
                display_frame_gpu = img_gpu_255.repeat(3, 1, 1)
                display_frame_gpu[0, outline_mask_bool_gpu] = 0   # B 通道
                display_frame_gpu[1, outline_mask_bool_gpu] = 255 # G 通道
                display_frame_gpu[2, outline_mask_bool_gpu] = 0   # R 通道

                final_image_cpu = display_frame_gpu.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                cv2.imwrite('test/{}'.format(img_path[j].split('/')[-1]), final_image_cpu)


if __name__ == '__main__':
    main()

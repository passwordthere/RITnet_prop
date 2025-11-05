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
from utils import get_predictions
import cv2
# from line_profiler import LineProfiler 


IMAGE_DIR = 'dataset/test/images' 
BATCH_SIZE = 4
NUM_WORKERS = 4

# @profile
def main():    
    device=torch.device("cuda")
    model = model_dict['densenet']
    model  = model.to(device)
    filename = 'best_model.pkl'
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    print(f"模型的默认数据类型是: {next(model.parameters()).dtype}")

    test_set = InferenceDataset(root_dir=IMAGE_DIR)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    counter=0
    
    os.makedirs('test/labels/',exist_ok=True)
    os.makedirs('test/output/',exist_ok=True)
    os.makedirs('test/mask/',exist_ok=True)
    
    with torch.no_grad():
        for i, batchdata in enumerate(testloader):
            img, img_path = batchdata
            data = img.to(device)       
            print(data.shape)
            output = model(data)            
            predict = get_predictions(output)
            for j in range(len(batchdata[0])):  
                inp_gpu = img[j].squeeze() * 0.5 + 0.5 
                img_orig_gpu = torch.clamp(inp_gpu, 0, 1) # [0, 1] 范围
                pred_gpu = predict[j].squeeze() / 3.0 # [0, 1] 范围
                combine_gpu = torch.hstack([img_orig_gpu, pred_gpu]) 
                display_frame = (combine_gpu * 255.0).byte().cpu().numpy() 
                cv2.imwrite('test/mask/{}'.format(img_path[j].split('/')[-1]), display_frame)


if __name__ == '__main__':
    main()

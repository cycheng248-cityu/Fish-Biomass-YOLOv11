# -*- coding: utf-8 -*-
"""
@author: Peter
"""

import os
import yaml
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import pandas as pd
import math
# from pathlib import PathSD
from types import SimpleNamespace
# import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.metrics import bbox_iou

#%%

# ==========================================
# DEFAULT SETTING: CIoU + OKS
# ==========================================
if __name__ == "__main__":
    DATASET_DIR = "C:/Users/user/Documents/fish_size_estimation"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    args = dict(
        model="yolo11n-pose.pt",  
        data=DATA_YAML,
        epochs=150,               
        imgsz=1280,               # Changed to 1280 for the manuscript math
        optimizer='SGD',        # Explicitly setting the optimizer
        cos_lr=False,          # <--- Turns on the smooth cosine decay
        lr0=0.01,             # Initial learning rate
        lrf=0.01,
        multi_scale=False,        # Turn off multi-scale so it stays strictly at 1280
        batch=16,
        device=device,                 
        
        # --- AUGMENTATION & LOSS GAINS (Keep these identical across all runs) ---
        mosaic=1.0,   
        close_mosaic=15,            
        mixup=0.1,                
        degrees=10.0,             
        scale=0.5,                
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        box=8.5, pose=12.0, kobj=2.0,
        warmup_epochs=5, augment=True, save=True
    )

    # Use the STANDARD YOLO class, not the FishPoseTrainer
    model = YOLO("yolo11n-pose.pt")
    model.train(**args)

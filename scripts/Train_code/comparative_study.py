# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 09:22:46 2026

@author: user
"""

import os
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    DATASET_DIR = "C:/Users/user/Documents/fish_size_estimation"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
    
    # The competitor models
    competitors = ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolo11n-pose.pt", "yolo11s-pose.pt"]

    # The EXACT SAME augmentations you used for your proposed model
    shared_args = dict(
        data=DATA_YAML,
        epochs=150,               
        imgsz=1280,               
        batch=8,                  
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu',                 
        
        # Standardize the optimizer for fairness
        optimizer='SGD',
        cos_lr=False,              
        lr0=0.01, # Use the optimal baseline LR you found in Config B
        lrf=0.01,
        patience=50,
        
        # Exact same augmentations
        mosaic=1.0,               
        close_mosaic=15,          
        mixup=0.1,                
        degrees=10.0,             
        scale=0.5,                
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        
        box=8.5, 
        pose=12.0, 
        kobj=2.0,
        
        warmup_epochs=5,
        augment=True,  
        save=True
    )

    for model_name in competitors:
        print(f"\n{'='*50}")
        print(f" STARTING FAIR BASELINE FOR: {model_name}")
        print(f"{'='*50}\n")
        
        torch.cuda.empty_cache()
        model = YOLO(model_name)
        
        # Run training using the shared arguments
        model.train(
            project="runs/comparative_study",
            name=model_name.replace(".pt", "_baseline"),
            **shared_args

        )

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
# CUSTOM LOSS: WIoU v3 + MSE
# ==========================================
class FishCustomLoss(v8PoseLoss):
    def __init__(self, model):
        if not hasattr(model, 'args') or model.args is None:
            model.args = SimpleNamespace(tal_topk=10)
        super().__init__(model)
        
        # Robustly get 'nc'
        self.nc = getattr(model, 'nc', getattr(model.model[-1], 'nc', 1))
        self.mse_kpt = nn.MSELoss(reduction='none')

    def kpts_loss(self, pred_kpts, gt_kpts, kpt_mask, area):
        """MSE for strict keypoint alignment."""
        dist_sq = (pred_kpts[..., 0:2] - gt_kpts[..., 0:2]).pow(2).sum(-1)
        return (dist_sq * kpt_mask).sum() / (kpt_mask.sum() + 1e-7)

    def calculate_wiou_v3(self, pbox, gbox, iou):
        """Wise IoU v3 for high-precision bounding boxes."""
        px, py, pw, ph = pbox.chunk(4, -1)
        gx, gy, gw, gh = gbox.chunk(4, -1)
        dist_sq = (px - gx).pow(2) + (py - gy).pow(2)
        cw = torch.max(px + pw/2, gx + gw/2) - torch.min(px - pw/2, gx - gw/2)
        ch = torch.max(py + ph/2, gy + gh/2) - torch.min(py - ph/2, gy - gh/2)
        
        wiou_v1 = torch.exp(dist_sq / (cw.pow(2) + ch.pow(2) + 1e-7).detach()) * (1.0 - iou)
        with torch.no_grad():
            beta = wiou_v1 / (wiou_v1.mean() + 1e-7)
            # Gradient modifier for dynamic focusing
            gradient_modifier = beta / (1 * (2 ** (beta - 2)))
        return gradient_modifier * wiou_v1

    def bbox_loss(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=False)
        loss_iou = self.calculate_wiou_v3(pred_bboxes[fg_mask], target_bboxes[fg_mask], iou)
        return (loss_iou * target_scores.sum(-1)[fg_mask].unsqueeze(-1)).sum() / target_scores_sum

# ==========================================
# CUSTOM TRAINER
# ==========================================
class FishPoseTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.args = self.args 
        model.criterion = FishCustomLoss(model)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.args = self.args
        self.model.criterion = FishCustomLoss(self.model)

# ==========================================
# MAIN TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    DATASET_DIR = "C:/Users/user/Documents/fish_size_estimation"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    args = dict(
        model="yolo11n-pose.pt",  
        data=DATA_YAML,
        epochs=150,               
        imgsz=1280,               
        multi_scale=False,        
        batch=16,                  
        
        # --- OPTIMIZER & SCHEDULER ---
        optimizer='AdamW',
        cos_lr=True,              
        lr0=0.003,
        lrf=0.01,
        
        # --- AUGMENTATION ---
        mosaic=1.0,               
        close_mosaic=15,          
        mixup=0.1,                
        degrees=10.0,             
        scale=0.5,                
        fliplr=0.5,
        hsv_h=0.015, 
        hsv_s=0.4, 
        hsv_v=0.4,
        
        # --- LOSS GAINS ---
        box=8.5, 
        pose=12.0, 
        kobj=2.0,
        
        warmup_epochs=5,
        augment=True,  
        save=True
    )

    trainer = FishPoseTrainer(overrides=args)
    trainer.train()
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
#  CUSTOM LOSS: Alpha-CIoU + Wing Loss
# ==========================================
class FishCustomLossv2(v8PoseLoss):
    def __init__(self, model):
        if not hasattr(model, 'args') or model.args is None:
            model.args = SimpleNamespace(tal_topk=10)
        super().__init__(model)
        
        # Robustly get 'nc'
        self.nc = getattr(model, 'nc', getattr(model.model[-1], 'nc', 1))

        # --- Wing Loss Parameters ---
        self.w = 10.0
        self.eps = 2.0

    # ------------------------------------------
    # KEYPOINT LOSS: WING LOSS
    # ------------------------------------------
    def wing_loss(self, diff):
        # diff: absolute error, e.g. |pred - gt|
        w, eps = self.w, self.eps
        C = w - w * torch.log1p(w / eps)
        small = diff < w
        loss = torch.where(
            small,
            w * torch.log1p(diff / eps),
            diff - C
        )
        return loss

    def kpts_loss(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Wing loss on keypoint coordinates for strict pixel alignment."""
        # Coordinate difference [N, K, 2]
        diff = (pred_kpts[..., 0:2] - gt_kpts[..., 0:2]).abs()  
        loss_coord = self.wing_loss(diff)                      
        loss_coord = loss_coord.sum(-1)                        # Sum over x, y
        loss_coord = loss_coord * kpt_mask                     # Mask invisible keypoints

        return loss_coord.sum() / (kpt_mask.sum() + 1e-7)

    # ------------------------------------------
    # BOUNDING BOX LOSS: ALPHA-CIoU
    # ------------------------------------------
    def calculate_alpha_ciou(self, pbox, gbox, alpha=3.0):
        """Alpha-CIoU for high-precision bounding boxes (Default alpha=3)."""
        px1, py1, px2, py2 = pbox.chunk(4, -1)
        gx1, gy1, gx2, gy2 = gbox.chunk(4, -1)

        pw, ph = px2 - px1, py2 - py1
        gw, gh = gx2 - gx1, gy2 - gy1
        pxc, pyc = px1 + pw / 2, py1 + ph / 2
        gxc, gyc = gx1 + gw / 2, gy1 + gh / 2

        inter_x1 = torch.max(px1, gx1)
        inter_y1 = torch.max(py1, gy1)
        inter_x2 = torch.min(px2, gx2)
        inter_y2 = torch.min(py2, gy2)
        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        inter_area = inter_w * inter_h

        union_area = (pw * ph) + (gw * gh) - inter_area + 1e-7
        iou = inter_area / union_area

        cw = torch.max(px2, gx2) - torch.min(px1, gx1)
        ch = torch.max(py2, gy2) - torch.min(py1, gy1)
        c_diag_sq = cw.pow(2) + ch.pow(2) + 1e-7

        rho_sq = (pxc - gxc).pow(2) + (pyc - gyc).pow(2)

        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(gw / (gh + 1e-7)) - torch.atan(pw / (ph + 1e-7)), 2)

        with torch.no_grad():
            ciou_modifier = v / (1.0 - iou + v + 1e-7)

        # Alpha-CIoU Loss Formula
        iou_term = torch.pow(iou.clamp(min=1e-6), alpha)
        dist_term = torch.pow(rho_sq / c_diag_sq, alpha)
        ar_term = torch.pow(ciou_modifier * v, alpha)

        loss_alpha = 1.0 - iou_term + dist_term + ar_term
        return loss_alpha

    def bbox_loss(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        if fg_mask.sum() == 0:
            return pred_bboxes.new_tensor(0.0)
        loss_iou = self.calculate_alpha_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask], alpha=2.0).squeeze(-1)
        weights = target_scores.sum(-1)[fg_mask]  
        return (loss_iou * weights).sum() / (target_scores_sum + 1e-7)

# ==========================================
# 2. CUSTOM TRAINER
# ==========================================
class FishPoseTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.args = self.args 
        model.criterion = FishCustomLossv2(model)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.args = self.args
        self.model.criterion = FishCustomLossv2(self.model)

# ==========================================
# 3. MAIN TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    DATASET_DIR = "D:/1st_project_automated_fish_length_biomass_estimation/app/app_image_data"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    args = dict(
        model="yolo11n-pose.pt",  
        data=DATA_YAML,
        epochs=150,               
        imgsz=640,               
        multi_scale=False,        
        batch=8,      
        device=device,        
        
        # --- THE STABILIZED OPTIMIZER ---
        optimizer='SGD',
        cos_lr=False,              
        lr0=0.01,        
        lrf=0.01,
        
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

    trainer = FishPoseTrainer(overrides=args)
    trainer.train()

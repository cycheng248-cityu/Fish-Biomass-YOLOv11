# Edge-Deployable Stereo Vision for Fish Biomass Estimation 

[![Paper](https://img.shields.io/badge/Status-Under_Review-yellow.svg)](#)
[![Framework](https://img.shields.io/badge/YOLO-v11n--pose-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](#)

This repository contains the official code, trained model weights, and data links for the manuscript: **"Edge-Deployable Stereo Vision for Fish Biomass Estimation via Lightweight YOLOv11n-Pose and Dynamic Geometry"** (Submitted to *Information Processing in Agriculture*).

## Overview
A significant bottleneck in smart aquaculture is the reliance on high-cost specialized 3D sensors or heavy computational infrastructure. This project provides an **ultra-low-cost, edge-deployable alternative** that utilizes a dual-webcam architecture powered by a customized deep learning and geometric pipeline.

### Core Innovations
* **Edge-Optimized AI:** A customized YOLOv11n-pose network ($\alpha$-CIoU + Wing Loss) achieving sub-pixel anatomical keypoint extraction at **111.6 FPS** using only **5.5 MB** of memory and **6.7 GFLOPs**.
* **Dynamic Locomotion Modeling:** Utilizes a **2nd-degree Bézier curve** and distinct temporal statistical filters (10th, 30th, and 95th percentiles) to counter fluid spinal flexion and stereoscopic foreshortening.
* **Volumetric Error Absorption:** Derives a multi-species **Dynamic Shape Factor ($K$)** via L2-regularized Ridge Regression, natively absorbing systematic depth errors to achieve an overall mass estimation **MAPE of 8.64%** and an **$R^2$ of 0.92**.

---

## Repository Structure

* `/models/` — Contains the trained YOLOv11n-pose weights (`.pt`).
* `/scripts/` — Python scripts for:
  * Sub-pixel coordinate extraction and 2nd-degree Bézier curve interpolation.
  * Temporal statistical filtration (IQR and percentile extraction).
  * L2-regularized Ridge Regression for biomass calculation.
* `/data/` — Processed biometric extractions and final biomass estimation CSV results.

---

## Data Availability

To ensure full reproducibility, our raw training datasets and stereoscopic video samples are hosted on dedicated academic data repositories:

* **Training Dataset:** The 1,151 annotated images (with 5-point anatomical keypoints) used to train the YOLOv11n-pose model are publicly available on Roboflow Universe at: `https://universe.roboflow.com/fishbiometric/fish-tracking-z9jdq/dataset/47`
* **Test Video:** 1-minute dual-camera video clips of individual fish are available at: Fish-Biomass-YOLOv11/data

---

## Pipeline Methodology

1. **Dual-View Inference:** The lightweight YOLOv11n-pose model tracks the snout, midline, and tail in the Top-View (X-Y plane) and Side-View (Z depth) simultaneously.
2. **Optical Correction:** Paraxial approximations and Snell's law ($n_{water}=1.33$, $n_{glass}=1.52$) correct baseline multi-media refraction.
3. **Geometric Extraction:** Bézier curve interpolation calculates the standard length, while statistical arrays extract true body thickness and height.
4. **Biomass Regression:** An ellipsoidal volume approximation is mapped to true physical mass using the dimensionless Dynamic Shape Factor ($K$).

---

## Citation
If you utilize this code or dataset in your research, please cite our forthcoming paper:

```bibtex
@article{cheng2026edge,
  title={Edge-Deployable Stereo Vision for Fish Biomass Estimation via Lightweight YOLOv11n-Pose and Dynamic Geometry},
  author={Cheng, Cheuk Yiu and Zhang, Yuxuan and Cai, Wenlong and Lau, Condon},
  journal={Information Processing in Agriculture},
  year={2026},
  note={Under Review}
}

---

## Contact
For any inquiries regarding the code or dataset, please contact Cheuk Yiu CHENG (cycheng248-c@my.cityu.edu.hk) at the Department of Physics, City University of Hong Kong.


# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 07:33:20 2026

@author: user
"""

import os
import time
import torch
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name, weight_path, imgsz=1280, iterations=100):
    print(f"\n{'='*50}")
    print(f" BENCHMARKING: {model_name}")
    print(f"{'='*50}")

    # 1. Physical Model Size (MB)
    file_size_mb = os.path.getsize(weight_path) / (1024 * 1024)
    print(f" Physical Size: {file_size_mb:.2f} MB")

    # Load Model
    model = YOLO(weight_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 2. Parameters & FLOPs
    # Ultralytics built-in info function
    model_info = model.model.info()
    params = model_info[1] / 1e6  # Convert to Millions
    flops = model_info[3]         # Already in GFLOPs
    print(f" Parameters:    {params:.2f} M")
    print(f" Complexity:    {flops:.1f} GFLOPs")

    # 3. Inference Latency & FPS
    # We must use a dummy image of the exact size you trained on (1280)
    print(f"\n Warming up GPU with imgsz={imgsz}...")
    dummy_input = torch.zeros((1, 3, imgsz, imgsz)).to(device)
    
    # Warmup (10 runs to wake up the GPU)
    for _ in range(10):
        model.predict(dummy_input, verbose=False)

    print(f"Running inference for {iterations} iterations...")
    start_time = time.time()
    
    # Actual Benchmark loop
    for _ in range(iterations):
        model.predict(dummy_input, verbose=False)
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps = 1000 / avg_latency_ms
    
    print(f" Avg Latency:   {avg_latency_ms:.2f} ms per frame")
    print(f" Throughput:    {fps:.1f} FPS")

    # 4. Total Training Time (Extracted from results.csv)
    # Assumes results.csv is in the same folder as the weights (e.g., runs/pose/train/weights/best.pt)
    run_dir = os.path.dirname(os.path.dirname(weight_path))
    results_path = os.path.join(run_dir, "results.csv")
    
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip() # Clean column names
        
        # YOLO results.csv has a 'time' column which is usually the time per epoch
        if 'time' in df.columns:
            total_train_seconds = df['time'].sum()
            hours = int(total_train_seconds // 3600)
            minutes = int((total_train_seconds % 3600) // 60)
            print(f" Train Time:    {hours} hours, {minutes} minutes (Total Epochs: {len(df)})")
        else:
            print(" 'time' column not found in results.csv")
    else:
        print(f" Could not find results.csv at {results_path}")

    print(f"{'='*50}\n")
    
    return {
        "Model": model_name,
        "Size (MB)": round(file_size_mb, 1),
        "Params (M)": round(params, 1),
        "GFLOPs": round(flops, 1),
        "Latency (ms)": round(avg_latency_ms, 2),
        "FPS": round(fps, 1)
    }

if __name__ == "__main__":
    # Update these paths tomorrow to point to your specific best.pt files!
    # For example: "runs/pose/yolo11n_baseline/weights/best.pt"
    models_to_test = {
        "YOLOv8n-pose": "C:/Users/user/Documents/fish_size_estimation/runs/comparative_study/yolov8n-pose_baseline/weights/best.pt",
        "YOLOv8s-pose": "C:/Users/user/Documents/fish_size_estimation/runs/comparative_study/yolov8s-pose_baseline/weights/best.pt",
        "YOLOv11n-pose": "C:/Users/user/Documents/fish_size_estimation/runs/pose/CIoU+OKS+SGD+lr0.01/weights/best.pt",
        "YOLOv11s-pose": "C:/Users/user/Documents/fish_size_estimation/runs/comparative_study/yolo11s-pose_baseline/weights/best.pt"
    }

    results_list = []
    
    for name, path in models_to_test.items():
        if os.path.exists(path):
            res = benchmark_model(name, path, imgsz=1280)
            results_list.append(res)
        else:
            print(f" Weights not found for {name} at {path}. Skipping...")

    # Print a beautiful summary table at the end
    if results_list:
        summary_df = pd.DataFrame(results_list)
        print("\n FINAL BENCHMARK SUMMARY ")
        print(summary_df.to_markdown(index=False))
        
        summary_df.to_csv("benchmark_summary.csv", index=False)
        print("\n Success: Results saved to 'benchmark_summary.csv'")

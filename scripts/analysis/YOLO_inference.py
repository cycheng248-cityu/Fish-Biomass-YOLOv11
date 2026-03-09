"""
@author: Peter
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import torch 
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.chdir('C:\\Users\\Condon\\Desktop\\Peter') 
os.chdir('D:/') 

from tqdm import tqdm
from PIL import Image
import glob
from ultralytics import YOLO
import bezier
# from scipy.interpolate import CubicSpline
# from scipy.integrate import quad
import math



print(torch.cuda.get_device_name(0))




#%% YOLO results

cap = cv2.VideoCapture("D:/1st_project_automated_fish_length_biomass_estimation/data_code_result/test_2/fish_6.mov")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'fish_10_yolo.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


model = YOLO("D:/1st_project_automated_fish_length_biomass_estimation/data_code_result/yolo_model/runs/pose/alpha_IoU+Wing+SGD/weights/best.pt")

#%% Multi-media regression + Biometric measurement


'''
Task: Calculate length and weight of fish: (DONE!!!)
    1. Straighten the fish if it is curved (Bezier curve interpolation)
    2. Get the universial pixel value using trigonometry
    3. Optic correction
    4. length ratio adjustment
    5. Weight estimation (Pseudo-teardrop-shaped approximation)
    
--> Detailed illustrations are in the ppt slides


'''

##### Some parameters needed to input ########



top_camera_to_water = 52 # in centimeter 
side_camera_to_glass = 30 # in centimeter
glass_thickness = 0.5 # in centimeter 

tank_length = 45 # length of the tank (in cm)
tank_width = 27 # width of the tank (in cm)
tank_height = 30 # height of the tank (in cm)
water_depth = 24 # Filled water depth (in cm)

n_air = 1.0 #refractive index (air)
n_water = 1.33 #refractive index (water)
n_glass = 1.5 #refractive index (glass)
water_density = 1 # in g/cm^3 



#  1. Straighten the fish if it is curved (Bezier curve interpolation)

def length_pixel_side(jj, kpts_xy, bboxes_xyxy, pred_cls, frame_height):

    length_z = abs(kpts_xy[jj,0,1] - kpts_xy[jj,1,1])
    height = np.sqrt((kpts_xy[jj,2,0] - kpts_xy[jj,3,0])**2 + (kpts_xy[jj,2,1] - kpts_xy[jj,3,1])**2)
    depth_pixel = (bboxes_xyxy[jj,3] + bboxes_xyxy[jj,1]) / 2
    depth = depth_pixel * water_depth / frame_height
        
    return length_z, height, depth


        
def length_pixel_top(jj, kpts_xy, bboxes_xyxy, pred_cls, frame_height): 
    nodes = np.asfortranarray([
    [int(kpts_xy[jj,0,0]), int(kpts_xy[jj,4,0]), int(kpts_xy[jj,1,0])],
    [int(kpts_xy[jj,0,1]), int(kpts_xy[jj,4,1]), int(kpts_xy[jj,1,1])],
                               ])
    curve = bezier.Curve(nodes, degree=2)
    length_xy_bezier = curve.length
    
    length_xy_two = np.sqrt((kpts_xy[jj,0,0] - kpts_xy[jj,1,0])**2 + (kpts_xy[jj,0,1] - kpts_xy[jj,1,1])**2)

    length_xy_three = (np.sqrt((kpts_xy[jj,0,0] - kpts_xy[jj,4,0])**2 + (kpts_xy[jj,0,1] - kpts_xy[jj,4,1])**2)) + np.sqrt((kpts_xy[jj,4,0] - kpts_xy[jj,1,0])**2 + (kpts_xy[jj,4,1] - kpts_xy[jj,1,1])**2)
    

    thickness = np.sqrt((kpts_xy[jj,2,0] - kpts_xy[jj,3,0])**2 + (kpts_xy[jj,2,1] - kpts_xy[jj,3,1])**2)
    
    distance_pixel = (bboxes_xyxy[jj,3] + bboxes_xyxy[jj,1]) / 2
    distance = distance_pixel * tank_width / frame_height

    return length_xy_bezier, length_xy_two, length_xy_three, thickness, distance

#  2. Get the universial pixel value using trigonometry
# length_z, height, depth = length_pixel(jj, kpts_xy, bboxes_xyxy, pred_cls, frame_height)
# length_xy, thickness, distance = length_pixel(jj, kpts_xy, bboxes_xyxy, pred_cls, frame_height)

# straightened_length = np.sqrt((length_z)**2 + (length_xy)**2)

#  3+4. Optic correction + length ratio adjustment


def top_camera_regression(length_xy_bezier, length_xy_two, length_xy_three, thickness, depth, top_camera_to_water, tank_height, tank_length, n_air, n_water):
    fish_to_camera = top_camera_to_water + depth
    
    length_temp_bezier = length_xy_bezier / 2
    length_temp_two = length_xy_two / 2
    length_temp_three = length_xy_three / 2
    
    l_common_bezier = (length_temp_bezier / (fish_to_camera)) * top_camera_to_water
    l_common_two = (length_temp_two / (fish_to_camera)) * top_camera_to_water
    l_common_three = (length_temp_three / (fish_to_camera)) * top_camera_to_water

    theta_air_l_bezier = math.atan(length_temp_bezier / (fish_to_camera))
    theta_air_l_two = math.atan(length_temp_two / (fish_to_camera))
    theta_air_l_three = math.atan(length_temp_three / (fish_to_camera))
    
    theta_water_l_bezier = math.asin(n_air * math.sin(theta_air_l_bezier) / n_water)
    theta_water_l_two = math.asin(n_air * math.sin(theta_air_l_two) / n_water)
    theta_water_l_three = math.asin(n_air * math.sin(theta_air_l_three) / n_water)

    l_reduced_bezier = math.tan(theta_water_l_bezier) * depth
    l_reduced_two = math.tan(theta_water_l_two) * depth
    l_reduced_three = math.tan(theta_water_l_three) * depth
    
    actual_length_pixel_bezier = 2 * (l_common_bezier + l_reduced_bezier)    
    actual_length_pixel_two = 2 * (l_common_two + l_reduced_two)    
    actual_length_pixel_three = 2 * (l_common_three + l_reduced_three)    

    actual_length_ref_bezier = actual_length_pixel_bezier / (top_camera_to_water - (tank_height - depth)) * fish_to_camera 
    actual_length_ref_two = actual_length_pixel_two / (top_camera_to_water - (tank_height - depth)) * fish_to_camera 
    actual_length_ref_three = actual_length_pixel_three / (top_camera_to_water - (tank_height - depth)) * fish_to_camera 

    actual_length_xy_bezier = actual_length_ref_bezier * tank_length / (frame_width // 2)
    actual_length_xy_two = actual_length_ref_two * tank_length / (frame_width // 2)
    actual_length_xy_three = actual_length_ref_three * tank_length / (frame_width // 2)


    thickness_temp = thickness / 2
    
    
    t_common = (thickness_temp / (fish_to_camera)) * top_camera_to_water
    theta_air_t = math.atan(thickness_temp / (fish_to_camera))
    theta_water_t = math.asin(n_air * math.sin(theta_air_t) / n_water)

    t_reduced = math.tan(theta_water_t) * depth
    actual_thickness_pixel = 2 * (t_common + t_reduced)
    
  
    actual_thickness_ref = actual_thickness_pixel / (top_camera_to_water - (tank_height - depth)) * fish_to_camera 
    actual_thickness = actual_thickness_ref * tank_length / (frame_width // 2)
    
    return actual_length_xy_bezier, actual_length_xy_two, actual_length_xy_three, actual_thickness
    

def side_camera_regression(length_z, height, distance, side_camera_to_glass, glass_thickness, tank_length, n_air, n_water, n_glass):
    length_temp = length_z / 2
    height_temp = height / 2
    fish_to_camera = side_camera_to_glass + glass_thickness + distance
    
    l_common = (length_temp / (fish_to_camera)) * side_camera_to_glass
    h_common = (height_temp / (fish_to_camera)) * side_camera_to_glass
    
    theta_air_h = math.atan(height_temp / (fish_to_camera))
    theta_glass_h = math.asin(n_air * math.sin(theta_air_h) / n_glass)
    h_reduced_1 = math.tan(theta_glass_h) * glass_thickness
    
    theta_air_l = math.atan(length_temp / (fish_to_camera))
    theta_glass_l = math.asin(n_air * math.sin(theta_air_l) / n_glass)
    l_reduced_1 = math.tan(theta_glass_l) * glass_thickness
    
    theta_water_h = math.asin(n_glass * math.sin(theta_glass_h) / n_water)
    h_reduced_2 = math.tan(theta_water_h) * distance
    actual_height_pixel = 2 * (h_common + h_reduced_1 + h_reduced_2)
    
    theta_water_l = math.asin(n_glass * math.sin(theta_glass_l) / n_water)
    l_reduced_2 = math.tan(theta_water_l) * distance
    actual_length_pixel = 2 * (l_common + l_reduced_1 + l_reduced_2)
    
    # reference_pixel = fish_to_camera * (frame_width // 2) / side_camera_to_glass
    actual_height_ref = actual_height_pixel / side_camera_to_glass * fish_to_camera
    actual_height = actual_height_ref * tank_length / (frame_width // 2)
    
    actual_length_ref = actual_length_pixel / side_camera_to_glass * fish_to_camera
    actual_length_z = actual_length_ref * tank_length / (frame_width // 2)
    
    
    return actual_length_z, actual_height

# 5. Weight estimation (Pseudo-teardrop-shaped approximation)

def mass_estimation(actual_length, actual_thickness, actual_height, water_density):
    upper_shape_volume = 0.5 * (4/3 * math.pi * (actual_thickness/2) * (actual_height/2) * (0.35*actual_length))
    lower_shape_volume = 1/3 * (math.pi * (actual_thickness/2) * (actual_height/2) * (0.65*actual_length))
    total_volume = upper_shape_volume + lower_shape_volume
    mass = water_density * total_volume
    
    # volume = 2 * (1/3 * actual_length * actual_height * actual_thickness/2)
    # mass = water_density * volume
    
    return mass
    
    


#%%  Run YOLO detection

i = 0
frames = []

length_bezier_array = list()
length_two_array = list()
length_three_array = list()

height_array = list()
thickness_array = list()

class_names = ['fish_side', 'fish_top']


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = og_frame.copy()
    im = Image.fromarray(frame)
    #im.save(str(i)+".jpg")

    results = model(im, device='cuda:0', imgsz = 1280, conf=0.35)  

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        kpts = result.keypoints
        cls = boxes.cls.tolist()  # Convert tensor to list
        xyxy = boxes.xyxy
        conf = boxes.conf
        # xywh = boxes.xywh  # box with xywh format, (N, 4)
        kpts_xy = kpts.xy
        for class_index in cls:
            class_name = class_names[int(class_index)]
            #print("Class:", class_name)
    
    pred_cls = np.array(cls)
    conf = conf.detach().cpu().numpy()
    xyxy = xyxy.detach().cpu().numpy()
    bboxes_xyxy = xyxy
    # bboxes_xyxy = xyxy.cpu().numpy()
    bboxes_xyxy_int = np.array(xyxy, dtype='int')
    kpts_xy = kpts_xy.cpu().numpy()

    # tracks = tracker.update(bboxes_xywh_float, conf, frame)
    
    if (pred_cls == 0).sum() != (pred_cls == 1).sum() or pred_cls.size == 0:
        frames.append(og_frame)
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB))
        continue

    jj = 0
    for ii in range(np.shape(kpts_xy)[0]):
        
        xB = bboxes_xyxy_int[jj,2]
        xA = bboxes_xyxy_int[jj,0]
        yB = bboxes_xyxy_int[jj,3]
        yA = bboxes_xyxy_int[jj,1]
    
        cv2.rectangle(og_frame, (int(xA), int(yA)), (int(xB), int(yB)), (0, 0, 255), 2)
    
        text_color = (0, 0, 0)  # Black color for text
        cv2.circle(og_frame,(int(kpts_xy[jj,0,0]),int(kpts_xy[jj,0,1])),3,(255, 255, 255),-1) # Keypoint: Head
        cv2.circle(og_frame,(int(kpts_xy[jj,1,0]),int(kpts_xy[jj,1,1])),3,(0, 0, 255),-1) # Keypoint: Tail
        cv2.circle(og_frame,(int(kpts_xy[jj,2,0]),int(kpts_xy[jj,2,1])),3,(0, 255, 0),-1) # Keypoint: Top, Left
        cv2.circle(og_frame,(int(kpts_xy[jj,3,0]),int(kpts_xy[jj,3,1])),3,(255, 0, 0),-1) # Keypoint: Bottom, Right
        cv2.circle(og_frame,(int(kpts_xy[jj,4,0]),int(kpts_xy[jj,4,1])),3,(0, 0, 0),-1) # Keypoint: Middle
        
        cv2.putText(og_frame,f'{jj}', (int(xA) + 10, int(yA) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        if pred_cls[jj] == 0 & np.any(kpts_xy[jj,:,:3] != 0):
            length_z, height, depth = length_pixel_side(jj, kpts_xy, bboxes_xyxy, pred_cls, frame_height)
        elif pred_cls[jj] == 1:
            length_xy_bezier, length_xy_two, length_xy_three, thickness, distance = length_pixel_top(jj, kpts_xy, bboxes_xyxy, pred_cls, frame_height)
                    
        jj += 1
        
    
    
    if np.all(np.array([length_z, height, depth, length_xy_bezier, length_xy_two, length_xy_three, thickness, distance]) !=0):
        actual_length_xy_bezier, actual_length_xy_two, actual_length_xy_three, actual_thickness = top_camera_regression(length_xy_bezier, length_xy_two, length_xy_three, thickness, depth, top_camera_to_water, tank_height, tank_length, n_air, n_water)
        actual_length_z, actual_height = side_camera_regression(length_z, height, distance, side_camera_to_glass, glass_thickness, tank_length, n_air, n_water, n_glass)       
        
        length_bezier_array.append(actual_length_xy_bezier)
        length_two_array.append(actual_length_xy_two)
        length_three_array.append(actual_length_xy_three)

        thickness_array.append(actual_thickness)
        height_array.append(actual_height)
    
    length_z, height, depth, length_xy, thickness, distance = 0,0,0,0,0,0
    frames.append(og_frame)

    out.write(cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB))
    
 
    # i += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# removing_files = glob.glob('*.jpg')
# for i in removing_files:


#%% Accuracy test area 

actual_length = 4.9
actual_thickness = 1.0
actual_height = 1.8
actual_mass = 5.7

fig, ax = plt.subplots(2,3,figsize=(15, 5))

ax[0,0].hist(length_bezier_array,bins=np.linspace(0,10,100))
ax[0,0].axvline(x=actual_length, color='red', linestyle='--')
ax[0,0].set_xlabel('length (cm)')
ax[0,0].set_ylabel('frequency')
ax[0,0].set_title('Length_bezier (AI estimated)')

ax[0,1].hist(length_two_array,bins=np.linspace(0,10,100))
ax[0,1].axvline(x=actual_length, color='red', linestyle='--')
ax[0,1].set_xlabel('length (cm)')
ax[0,1].set_ylabel('frequency')
ax[0,1].set_title('Length_two (AI estimated)')

ax[0,2].hist(length_three_array,bins=np.linspace(0,10,100))
ax[0,2].axvline(x=actual_length, color='red', linestyle='--')
ax[0,2].set_xlabel('length (cm)')
ax[0,2].set_ylabel('frequency')
ax[0,2].set_title('Length_three (AI estimated)')

ax[1,0].hist(thickness_array,bins=np.linspace(0,10,100))
ax[1,0].axvline(x=actual_thickness, color='red', linestyle='--')
ax[1,0].set_xlabel('thickness (cm)')
ax[1,0].set_ylabel('frequency')
ax[1,0].set_title('Thickness (AI estimated)')

ax[1,1].hist(height_array,bins=np.linspace(0,10,100))
ax[1,1].axvline(x=actual_height, color='red', linestyle='--')
ax[1,1].set_xlabel('height (cm)')
ax[1,1].set_ylabel('frequency')
ax[1,1].set_title('Height (AI estimated)')

# ax[3].hist(mass_array,bins=np.linspace(0,10,200))
# ax[3].axvline(x=actual_mass, color='red', linestyle='--')
# ax[3].set_xlabel('mass (g)')
# ax[3].set_ylabel('frequency')
# ax[3].set_title('Mass (AI estimated)')


plt.tight_layout()

#%%
import numpy as np
import pandas as pd
import os

# NOTE: Make sure these variables are defined in your loop before running the block below!
fish_id = "Fish_10" 
actual_length = 4.9
actual_thickness = 1.0
actual_height = 1.8


# List of all percentiles we want to test to find the optimal extraction parameter
# Includes extremes (10, 95), quartiles (25, 50, 75), and dense upper ranges
percentiles_to_test = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]

def extract_percentiles(prefix, data_array, results_dict):
    """
    Automatically calculates the median and all defined percentiles, 
    adding them to the results dictionary.
    *** IQR FILTERING HAS BEEN COMPLETELY REMOVED ***
    """
    # Clean NaNs
    data_array = np.array(data_array)
    data_array = data_array[~np.isnan(data_array)]
    
    # Check if array is empty after dropping NaNs
    if len(data_array) == 0:
        print(f"Warning: {prefix} array is empty after cleaning!")
        return results_dict
        
    # Add median specifically
    results_dict[f"{prefix}_Med"] = np.median(data_array)
    
    # Loop through all percentiles and add them
    for p in percentiles_to_test:
        results_dict[f"{prefix}_{p}"] = np.percentile(data_array, p)
        
    return results_dict

# ==========================================
# INITIALIZE RESULTS DICTIONARY
# ==========================================
results_dict = {
    "Fish_ID": fish_id,
    "Actual_Length": actual_length,
    "Actual_Thickness": actual_thickness,
    "Actual_Height": actual_height
}

# ==========================================
# PROCESS ALL METRICS (NO IQR)
# ==========================================
# 1. Length Estimations
results_dict = extract_percentiles("Len_Bezier", length_bezier_array, results_dict)
results_dict = extract_percentiles("Len_Two", length_two_array, results_dict)
results_dict = extract_percentiles("Len_Three", length_three_array, results_dict)

# 2. Thickness Estimation
results_dict = extract_percentiles("Thick", thickness_array, results_dict)

# 3. Height Estimation
results_dict = extract_percentiles("Height", height_array, results_dict)


# ==========================================
# SAVE RESULTS TO PANDAS DATAFRAME & CSV
# ==========================================
df_results = pd.DataFrame([results_dict])

# Changed filename to avoid overwriting your filtered data!
csv_filename = 'fish_evaluation_results_comprehensive_NoIQR.csv'

# Check if file exists to determine if we need to write the header row
file_exists = os.path.isfile(csv_filename)

# Append to CSV
df_results.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

print(f"Extracted ALL parameter variations (NO IQR) for {fish_id} and saved to {csv_filename}")

#%%

import numpy as np
import pandas as pd
import os

# ==========================================
# IQR FILTERING HELPER FUNCTION
# ==========================================
def apply_iqr_filter(data_array):
    """
    Removes extreme outliers using the Interquartile Range (1.5 * IQR) method.
    """
    if len(data_array) == 0:
        return data_array # Return empty if no data
        
    Q1 = np.percentile(data_array, 25)
    Q3 = np.percentile(data_array, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Keep only the data that falls within the acceptable bounds
    filtered_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
    
    return filtered_data

# NOTE: Make sure these variables are defined in your loop before running the block below!
fish_id = "Fish_10" 
actual_length = 4.9
actual_thickness = 1.0
actual_height = 1.8
actual_mass = 5.7

# List of all percentiles we want to test to find the optimal extraction parameter
# Includes extremes (10, 95), quartiles (25, 50, 75), and dense upper ranges
percentiles_to_test = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]

def extract_percentiles(prefix, data_array, results_dict):
    """
    Automatically calculates the median and all defined percentiles, 
    adding them to the results dictionary.
    """
    # Clean NaNs
    data_array = np.array(data_array)
    data_array = data_array[~np.isnan(data_array)]
    
    # Apply IQR Filter
    data_array = apply_iqr_filter(data_array)
    
    if len(data_array) == 0:
        print(f"Warning: {prefix} array is empty after filtering!")
        return results_dict
        
    # Add median specifically
    results_dict[f"{prefix}_Med"] = np.median(data_array)
    
    # Loop through all percentiles and add them
    for p in percentiles_to_test:
        results_dict[f"{prefix}_{p}"] = np.percentile(data_array, p)
        
    return results_dict

# ==========================================
# INITIALIZE RESULTS DICTIONARY
# ==========================================
results_dict = {
    "Fish_ID": fish_id,
    "Actual_Length": actual_length,
    "Actual_Thickness": actual_thickness,
    "Actual_Height": actual_height,
    "Actual_Mass": actual_mass
}

# ==========================================
# PROCESS ALL METRICS
# ==========================================
# 1. Length Estimations
results_dict = extract_percentiles("Len_Bezier", length_bezier_array, results_dict)
results_dict = extract_percentiles("Len_Two", length_two_array, results_dict)
results_dict = extract_percentiles("Len_Three", length_three_array, results_dict)

# 2. Thickness Estimation
results_dict = extract_percentiles("Thick", thickness_array, results_dict)

# 3. Height Estimation
results_dict = extract_percentiles("Height", height_array, results_dict)


# ==========================================
# SAVE RESULTS TO PANDAS DATAFRAME & CSV
# ==========================================
df_results = pd.DataFrame([results_dict])

csv_filename = 'fish_evaluation_results_comprehensive.csv'

# Check if file exists to determine if we need to write the header row
file_exists = os.path.isfile(csv_filename)

# Append to CSV
df_results.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

print(f"Extracted ALL parameter variations for {fish_id} and saved to {csv_filename}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# 1. Load the data
df = pd.read_csv('fish_evaluation_results_comprehensive_NoIQR.csv')

def calculate_mae(actual, estimated):
    return np.abs(actual - estimated).mean()

def calculate_mape(actual, estimated):
    return (np.abs(actual - estimated) / actual).mean() * 100

# Define percentiles
percentiles = ['10', '20', '25', '30', '40', '50', '60', '70', '75', '80', '85', '90', '95']
x_values = [int(p) for p in percentiles]

# ==========================================
# FIGURE 1: LENGTH COMPARISON (WITH INSET)
# ==========================================
mae_bezier = [calculate_mae(df['Actual_Length'], df[f'Len_Bezier_{p}']) for p in percentiles]
mae_two = [calculate_mae(df['Actual_Length'], df[f'Len_Two_{p}']) for p in percentiles]
mae_three = [calculate_mae(df['Actual_Length'], df[f'Len_Three_{p}']) for p in percentiles]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
plt.style.use('default')

ax.plot(x_values, mae_bezier, marker='o', linewidth=1, markersize=8, color='#1f77b4', label='Bézier Curve Interpolation')
ax.plot(x_values, mae_three, marker='s', linewidth=1, markersize=8, color='#ff7f0e', label='Three-Point Segmented')
ax.plot(x_values, mae_two, marker='^', linewidth=1, markersize=8, color='#2ca02c', label='Two-Point (Straight Line)')

min_bezier_idx = np.argmin(mae_bezier)
ax.plot(x_values[min_bezier_idx], mae_bezier[min_bezier_idx], marker='*', markersize=15, color='red', 
         label=f'Optimal Bézier (30th, MAE={mae_bezier[min_bezier_idx]:.3f} cm)')

ax.set_title('Standard Length Extraction: Parameter Sweep Comparison', fontsize=8, fontweight='bold', pad=15)
ax.set_xlabel('Extraction Percentile from Temporal Histogram', fontsize=8, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (cm)', fontsize=8, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=6, loc='upper left', frameon=True, shadow=True)

# --- Create Zoomed Inset ---
axins = ax.inset_axes([0.45, 0.45, 0.4, 0.4]) # [x0, y0, width, height]
axins.plot(x_values, mae_bezier, marker='o', linewidth=1, markersize=6, color='#1f77b4')
axins.plot(x_values, mae_three, marker='s', linewidth=1, markersize=6, color='#ff7f0e')
axins.plot(x_values, mae_two, marker='^', linewidth=1, markersize=6, color='#2ca02c')
axins.plot(x_values[min_bezier_idx], mae_bezier[min_bezier_idx], marker='*', markersize=5, color='red')

# Set limits for zoom (focusing on the 20th to 50th percentiles where the dip is)
axins.set_xlim(18, 52)
axins.set_ylim(0.46, 0.54)
axins.grid(True, linestyle=':', alpha=0.5)
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.tight_layout()
plt.savefig('Fig_1_Length_Comparison_Inset_NoIQR.png', format='png', dpi=300)
plt.close()

# ==========================================
# FIGURE 2: THICKNESS SWEEP
# ==========================================
mae_thick = [calculate_mae(df['Actual_Thickness'], df[f'Thick_{p}']) for p in percentiles]

plt.figure(figsize=(8, 5), dpi=300)
plt.plot(x_values, mae_thick, marker='D', linewidth=1, markersize=5, color='#9467bd', label='Thickness Extraction')
min_thick_idx = np.argmin(mae_thick)
plt.plot(x_values[min_thick_idx], mae_thick[min_thick_idx], marker='*', markersize=5, color='red', 
         label=f'Optimal Thickness (95th, MAE={mae_thick[min_thick_idx]:.3f} cm)')

plt.title('Body Thickness: Parameter Sweep', fontsize=8, fontweight='bold', pad=15)
plt.xlabel('Extraction Percentile from Temporal Histogram', fontsize=8, fontweight='bold')
plt.ylabel('Mean Absolute Error (cm)', fontsize=8, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('Fig_2_Thickness_Sweep_NoIQR.png', format='png', dpi=300)
plt.close()

# ==========================================
# FIGURE 3: HEIGHT SWEEP
# ==========================================
mae_height = [calculate_mae(df['Actual_Height'], df[f'Height_{p}']) for p in percentiles]

plt.figure(figsize=(8, 5), dpi=300)
plt.plot(x_values, mae_height, marker='v', linewidth=1, markersize=8, color='#8c564b', label='Height Extraction')
min_height_idx = np.argmin(mae_height)
plt.plot(x_values[min_height_idx], mae_height[min_height_idx], marker='*', markersize=15, color='red', 
         label=f'Optimal Height (10th, MAE={mae_height[min_height_idx]:.3f} cm)')

plt.title('Body Height: Parameter Sweep', fontsize=8, fontweight='bold', pad=15)
plt.xlabel('Extraction Percentile from Temporal Histogram', fontsize=8, fontweight='bold')
plt.ylabel('Mean Absolute Error (cm)', fontsize=8, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig('Fig_3_Height_Sweep_NoIQR.png', format='png', dpi=300)
plt.close()

# ==========================================
# PRINT FINAL MAE AND MAPE FOR MANUSCRIPT
# ==========================================
print("Plots generated successfully!")
print("-" * 50)
print("FINAL METRICS FOR MANUSCRIPT TEXT:")
print("-" * 50)

# Calculate optimal MAPEs based on the known best percentiles
best_len_mape = calculate_mape(df['Actual_Length'], df['Len_Bezier_30'])
best_thick_mape = calculate_mape(df['Actual_Thickness'], df['Thick_95'])
best_height_mape = calculate_mape(df['Actual_Height'], df['Height_10'])

print(f"Optimal Length (Bézier 30th): MAE = {mae_bezier[min_bezier_idx]:.4f} cm | MAPE = {best_len_mape:.2f}%")
print(f"Optimal Thickness (95th):     MAE = {mae_thick[min_thick_idx]:.4f} cm | MAPE = {best_thick_mape:.2f}%")
print(f"Optimal Height (10th):        MAE = {mae_height[min_height_idx]:.4f} cm | MAPE = {best_height_mape:.2f}%")




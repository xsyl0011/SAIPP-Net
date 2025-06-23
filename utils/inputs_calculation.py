#%%
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import numba as nb

# 3GPP InH
def path_loss_LOS(distance_matrix):
    LOS = np.zeros_like(distance_matrix)
    mask1 = distance_matrix >= 1
    LOS[mask1] = 32.4 + 17.3 * np.log10(distance_matrix[mask1]) + 20 * np.log10(0.868)
    mask2 = distance_matrix < 1
    LOS[mask2] = 27.0
    return LOS

def path_loss_NLOS(distance_matrix):
    NLOS = np.zeros_like(distance_matrix)
    mask = distance_matrix >= 1
    NLOS[mask] = 17.3 + 38.3 * np.log10(distance_matrix[mask]) + 24.9 * np.log10(0.868)
    mask2 = distance_matrix < 1
    NLOS[mask2] = 27.0
    return NLOS

# 使用numba加速Bresenham算法
@nb.njit
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

@nb.njit(parallel=True)
def check_line_values_add(matrix, x_ant, y_ant):
    h, w = matrix.shape[0], matrix.shape[1]
    los_map = np.zeros_like(matrix, dtype=nb.float32)
    
    for x1 in nb.prange(w):
        for y1 in range(h):
            points = bresenham(x_ant, y_ant, x1, y1)
            add = 0.0
            for point in points:
                x, y = point
                if x < w and y < h and x >= 0 and y >= 0:
                    add += float(matrix[y, x])
            los_map[y1, x1] = add
    
    return los_map

@nb.njit(parallel=True)
def check_line_values_count(matrix, x_ant, y_ant):
    h, w = matrix.shape[0], matrix.shape[1]
    los_map = np.zeros_like(matrix, dtype=nb.float32)
    
    value_list = np.array([1,2,3,4,6,10,23], dtype=np.float32)
    
    for x1 in nb.prange(w):
        for y1 in range(h):
            points = bresenham(x_ant, y_ant, x1, y1)
            count = 0
            for point in points:
                x, y = point
                if x < w and y < h and x >= 0 and y >= 0:
                    pixel_value = float(matrix[y, x])
                    for val in value_list:
                        if abs(pixel_value - val) < 0.1:
                            count += 1
                            break
            los_map[y1, x1] = count
    
    return los_map

@nb.njit(parallel=True)
def check_line_material_counts(matrix, x_ant, y_ant):
    h, w = matrix.shape[0], matrix.shape[1]
    
    map_1 = np.zeros_like(matrix, dtype=nb.float32)
    map_2 = np.zeros_like(matrix, dtype=nb.float32)
    map_3 = np.zeros_like(matrix, dtype=nb.float32)
    map_4 = np.zeros_like(matrix, dtype=nb.float32)
    map_6 = np.zeros_like(matrix, dtype=nb.float32)
    map_10 = np.zeros_like(matrix, dtype=nb.float32)
    map_23 = np.zeros_like(matrix, dtype=nb.float32)
    
    for x1 in nb.prange(w):
        for y1 in range(h):
            points = bresenham(x_ant, y_ant, x1, y1)
            count_1, count_2, count_3 = 0, 0, 0
            count_4, count_6, count_10, count_23 = 0, 0, 0, 0
            
            for point in points:
                x, y = point
                if x < w and y < h and x >= 0 and y >= 0:
                    pixel_value = float(matrix[y, x])
                    
                    if abs(pixel_value - 1) < 0.1:
                        count_1 += 1
                    elif abs(pixel_value - 2) < 0.1:
                        count_2 += 1
                    elif abs(pixel_value - 3) < 0.1:
                        count_3 += 1
                    elif abs(pixel_value - 4) < 0.1:
                        count_4 += 1
                    elif abs(pixel_value - 6) < 0.1:
                        count_6 += 1
                    elif abs(pixel_value - 10) < 0.1:
                        count_10 += 1
                    elif abs(pixel_value - 23) < 0.1:
                        count_23 += 1
            
            map_1[y1, x1] = count_1
            map_2[y1, x1] = count_2
            map_3[y1, x1] = count_3
            map_4[y1, x1] = count_4
            map_6[y1, x1] = count_6
            map_10[y1, x1] = count_10
            map_23[y1, x1] = count_23
    
    return map_1, map_2, map_3, map_4, map_6, map_10, map_23

def split_name(file_name):
    split = file_name.split("_")
    b_idx = int(split[0][1:])
    ant_idx = int(split[1][-1])
    f_idx = int(split[2][-1])
    s_idx = int(split[3][1:])
    return b_idx, ant_idx, f_idx, s_idx


def inputs_calculation(filename, HW):
    b_idx, ant_idx, f_idx, s_idx = split_name(filename)

    Sampling_positions = pd.read_csv(f"MLSP25_Test_Data/Test_Data_Positions/Positions_B" +  str(b_idx) +  "_Ant"+ str(ant_idx) + "_f"  + str(f_idx) + '.csv')
    # Sampling_positions = pd.read_csv(f"E:\MLSP\Evaluation_Data_T3\Positions\Positions_B" +  str(b_idx) +  "_Ant"+ str(ant_idx) + "_f"  + str(f_idx) + '.csv')
    
    # Building_Details = pd.read_csv(f"MLSP25_Test_Data/Test_Building_Details/T_B" +  str(b_idx) +  "_Details.csv")
    # W, H = Building_Details["W"].iloc[0], Building_Details["H"].iloc[0]
    
    x_ant = Sampling_positions["Y"].loc[s_idx]
    y_ant = Sampling_positions["X"].loc[s_idx]
    
    # num_calculations += 1

    file_name = f"B{b_idx}_Ant1_f1_S0_green.png"
    image_path = os.path.join(f'MLSP25_Test_Data\RGB', file_name)
    
    img = np.array(Image.open(image_path), dtype=np.float32)/10

    map_1, map_2, map_3, map_4, map_6, map_10, map_23 = check_line_material_counts(img, int(x_ant), int(y_ant))
    material_maps = np.stack((map_1, map_2, map_3, map_4, map_6, map_10, map_23), axis=0)
    material_maps = np.clip(material_maps, 0, 5)

    material_maps_transposed = np.transpose(material_maps, (1, 2, 0))
    material_maps_resized = cv2.resize(material_maps_transposed, (256, 256), interpolation=cv2.INTER_CUBIC)
    material_maps = np.transpose(material_maps_resized, (2, 0, 1))

    H, W = int(HW[0]), int(HW[1])
    X_points = np.repeat(np.linspace(0, W-1, W), H, axis=0).reshape(W, H).transpose()
    Y_points = np.repeat(np.linspace(0, H-1, H), W, axis=0).reshape(H,W)
    distance_matrix = np.sqrt((X_points - x_ant)**2 + (Y_points - y_ant)**2) * 0.25
    
    # calculate the model channel
    path_loss_LOS_matrix = path_loss_LOS(distance_matrix)
    path_loss_NLOS_matrix = path_loss_NLOS(distance_matrix)
    path_loss_matrix = np.zeros_like(distance_matrix)
    los_map = map_1 + map_2 + map_3 + map_4 + map_6 + map_10 + map_23
    los_map = np.clip(los_map, None, 5)
    path_loss_matrix = np.where(los_map == 0, path_loss_LOS_matrix, np.maximum(path_loss_LOS_matrix, path_loss_NLOS_matrix))
    normalized_path_loss_8bit = path_loss_matrix.astype(np.uint8)

    return material_maps, normalized_path_loss_8bit, los_map

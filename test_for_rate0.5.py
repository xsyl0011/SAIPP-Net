import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import pandas as pd
import time
from lib import test_dataloader
import os

# task = 1 for random sampling, task = 2 for designed sampling, please choose one of them
task = 1

input_path =  f"MLSP25_Test_Data/Inputs/Task_1/"

# only for task 1
samples_path = f"samples_for_task1/rate0.5/sampledGT/"

# only for task 2
sparse_mask_path = f"sample_locations_for_task2/rate0.5/"
ground_truth_path = f"ground_truth/"

# choose the model to be tested accrording to the task
# rmse = '0.011359' for task 1, rmse = '0.011093' for task 2
if task == 1:
    rmse = '0.011359'
elif task == 2:
    rmse = '0.011093'

# inferred radio maps saving path
save_path = f"inferred_PL_radiomaps/task{task}/rate0.5/"
os.makedirs(save_path, exist_ok=True)

Input_list_file_names = []

for b in [1,2,4,5,6]:
    if b == 1 or b == 5:
        for s in range (25,50):
            Input_list_file_names.append("B" + str(b) +  "_Ant"+  str(1) + "_f"  + str(1) + "_S" +str(s))
    else:
        for s in range (0,50):
            Input_list_file_names.append("B" + str(b) +  "_Ant"+  str(1) + "_f"  + str(1) + "_S" +str(s))


Input_list_IDs =  np.arange(0, len(Input_list_file_names), 1, dtype=int)

test_IDs = Input_list_IDs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim_X = 256
dim_y = 256
n_channels = 6
test_generator = test_dataloader.DataGenerator(
    task=task,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    list_IDs=test_IDs,
    file_names=Input_list_file_names,
    input_path=input_path,
    samples_path=samples_path,
    sparse_mask_path=sparse_mask_path,
    ground_truth_path = ground_truth_path,
    dim_X=dim_X,
    dim_y=dim_y,
    n_channels=n_channels,
    batch_size=1,
    shuffle=False
)

from lib import modules256Net_f_deltaN_samples
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = modules256Net_f_deltaN_samples.RadioNet(inputs=6)
model.load_state_dict(torch.load(f"trained_models/Trained_Model_{rmse}.pt"))
model.to(device)


def test_loss(model):
    print("Testing...")
    
    solution = pd.DataFrame(columns=["ID", "PL (dB)"])
    model.eval()
    index = -1
    
    total_processing_time = 0
    
    dataloader = DataLoader(test_generator, batch_size=1, shuffle=False, 
                          generator=torch.Generator(device='cpu'))
    
    # ---------- warm-up ----------
    print("warm-up...")

    warmup_dataloader = DataLoader(test_generator, batch_size=1, shuffle=False,
                                generator=torch.Generator(device='cpu'))
    warmup_iter = iter(warmup_dataloader)

    for i in range(3):
        warmup_inputs, _ = next(warmup_iter)
        warmup_inputs = warmup_inputs.to(device)
        
        with torch.no_grad():
            _ = model(warmup_inputs)
            torch.cuda.synchronize()
    
    print("warm-up done.")
    # --------------------------------------
    
    dataloader_iter = iter(dataloader)
    total_samples = len(dataloader)
    
    while index < total_samples - 1:
        index += 1
        
        # begin timing
        start_time = time.time()
        
        try:
            inputs, HW = next(dataloader_iter)
        except StopIteration:
            break
            
        inputs = inputs.to(device)
        
        # model inference
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            
            # post-processing
            outputs = outputs.squeeze().cpu().numpy()
            width = int(HW[0,1].item())
            height = int(HW[0,0].item())
            outputs1 = cv2.resize(outputs, (width, height), cv2.INTER_CUBIC)
            
            # convert to dB
            outputs1 = outputs1 * 255
            
            # clip the values to the range [13, 160]
            outputs1 = np.clip(outputs1, 13, 160)
            
            # end timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_processing_time += elapsed_time
        
        name = Input_list_file_names[index]
        # save as grayscale images
        cv2.imwrite(os.path.join(save_path, name + ".png"), outputs1)
        
        # save as pandas DataFrame
        y_flat = np.expand_dims(outputs1.flatten(), 1)
        y_names = np.expand_dims(np.core.defchararray.add(name + "_", 
                    np.linspace(0, outputs1.size-1, outputs1.size).astype(int).astype(str)), 1)             
        y_data = np.concatenate((y_names, y_flat), axis=1)
        y_pd = pd.DataFrame(data=y_data, index=np.linspace(0, outputs1.size-1, outputs1.size).astype(int), 
                           columns=["ID", "PL (dB)"]) 
        solution = pd.concat([solution, y_pd], ignore_index=True)
        
        # print the time taken for each sample
        print(f"{name}-Sample {index+1}/{total_samples}: {elapsed_time*1000:.2f}ms")
    
    # calculate the average time taken for all samples
    average_time = total_processing_time / (index + 1) * 1000  # convert to ms
    print(f"{index+1} samples have been processed.")
    print(f"Average run-time: {average_time:.2f}ms")
    
    return solution

solution = test_loss(model)
solution.to_csv(f'Solution_check_task{task}_{rmse}_rate0.5.csv', index=False)

#%% use 0.02% samples (for both task 1 and task 2) to finetune the pretrained model
import torch
import numpy as np
from lib import finetune_dataloader
import torch
from torch.utils.data import DataLoader
import cv2
import copy
import numpy as np
import pandas as pd
from skimage.io import imread
import time
import os

task = 1  # task = 1 for random sampling, task = 2 for designed sampling, please choose one of them

config = {
    'task': task,
    'rate': 0.02,
    'model': 'delta',
    'resolution': 256,
    'n_channels': 5,
    'augment': False,
    'learning_rate': 0.0001,
    'batch_size': 8,
    'num_epochs': 2,
    'train_shuffle': True,
    'pre_training': True,
    'pre_train_model': '0.017052',
    'new_network': True,
}

input_path =  f"MLSP25_Test_Data/Inputs/Task_1/"
# input_path =  "E:\MLSP\Evaluation_Data_T1\Inputs\Task_1/"

# only for task 1
samples_path = f"samples_for_task1/rate0.02/sampledGT/"
# samples_path = "E:\MLSP\\rate" + str(config['rate']) + "\sampledGT/"

# only for task 2
sparse_mask_path = f"sample_locations_for_task2/rate0.02/"
ground_truth_path = f"ground_truth/"

# inferred radio maps saving path
save_path = f"inferred_PL_radiomaps/task{task}/rate0.02/"
os.makedirs(save_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Input_list_file_names = []

# for b in [1,5]: 
#     for s in range (25):
#         Input_list_file_names.append("B" + str(b) +  "_Ant"+  str(1) + "_f"  + str(1) + "_S" +str(s))

for b in [1,2,4,5,6]:
    if b == 1 or b == 5:
        for s in range (25,50):
            Input_list_file_names.append("B" + str(b) +  "_Ant"+  str(1) + "_f"  + str(1) + "_S" +str(s))
    else:
        for s in range (0,50):
            Input_list_file_names.append("B" + str(b) +  "_Ant"+  str(1) + "_f"  + str(1) + "_S" +str(s))

Input_list_IDs =  np.arange(0, len(Input_list_file_names), 1, dtype=int)

np.random.seed(42)
np.random.shuffle(Input_list_IDs)

train_size = int(1 * len(Input_list_IDs))
train_IDs = Input_list_IDs[:train_size]
val_IDs = train_IDs
# val_IDs = Input_list_IDs[train_size:]
# print(Input_list_IDs)
# print(Input_list_file_names)
# print(Input_list_IDs.size)

#%% data generator
dim_X=config['resolution']
dim_y=config['resolution']
n_channels=config['n_channels']

train_generator = finetune_dataloader.DataGenerator(
    task=task,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    list_IDs=train_IDs,
    file_names=Input_list_file_names,
    input_path=input_path,
    samples_path=samples_path,
    sparse_mask_path=sparse_mask_path,
    ground_truth_path = ground_truth_path,
    dim_X=dim_X,
    dim_y=dim_y,
    n_channels=n_channels,
    batch_size=1,
    shuffle=config['train_shuffle'],
    augment=False
)

val_generator = finetune_dataloader.DataGenerator(
    task=task,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    list_IDs=val_IDs,
    file_names=Input_list_file_names,
    input_path=input_path,
    samples_path=samples_path,
    sparse_mask_path=sparse_mask_path,
    ground_truth_path = ground_truth_path,
    dim_X=dim_X,
    dim_y=dim_y,
    n_channels=n_channels,
    batch_size=1,
    shuffle=False,
    augment=False
)

dataloaders = {
    'train': DataLoader(train_generator, batch_size=config['batch_size'], shuffle=config['train_shuffle'], generator=torch.Generator(device = 'cuda')),
    'val': DataLoader(val_generator, batch_size=config['batch_size'], shuffle=False, generator=torch.Generator(device = 'cuda'))
}
#%% load and modify the weights of the pretrained model
def load_legacy_weights(model, pretrained_path):

    state_dict = torch.load(pretrained_path)
    
    if 'material_weights' not in state_dict and 'delta_1' in state_dict:
        delta_values = [
            state_dict['delta_1'],
            state_dict['delta_2'],
            state_dict['delta_3'], 
            state_dict['delta_4'],
            state_dict['delta_6'],
            state_dict['delta_10'],
            state_dict['delta_23']
        ]
        
        material_weights = torch.stack(delta_values)
        
        state_dict['material_weights'] = material_weights
        
        for key in ['delta_1', 'delta_2', 'delta_3', 'delta_4', 
                    'delta_6', 'delta_10', 'delta_23']:
            if key in state_dict:
                del state_dict[key]
    
    model.load_state_dict(state_dict, strict=False)
    return model

#%% load model
from lib import modules256Net_f_deltaN
import torch

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled

model = modules256Net_f_deltaN.RadioNet(inputs=6)

model.cuda()

if config['pre_training'] and config['new_network']:
    load_legacy_weights(model, f"trained_models/Trained_Model_{config['pre_train_model']}.pt")
elif config['pre_training'] and not config['new_network']:
    model.load_state_dict(torch.load(f"trained_models/Trained_Model_{config['pre_train_model']}.pt"))

#%% finetuning loop

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

def calc_loss_sparse(pred, samples, masks, metrics):
    loss_mse = torch.sum(((pred * masks - samples * masks) ** 2)) / torch.sum(masks)
    loss_rmse = torch.sqrt(loss_mse)
    # loss_rmse = torch.sqrt(loss_mse)
    metrics['mse'] += loss_mse.item() * samples.size(0)
    metrics['rmse'] += loss_rmse.item() * samples.size(0)
    return loss_rmse

def print_metrics(metrics, epoch_samples, phase):
    outputs = ""
    for k in metrics.keys():
        outputs += "{}: {:4f}, ".format(k, metrics[k] / epoch_samples)
    outputs = outputs.rstrip(", ")
    print("{}: {}".format(phase, outputs))


def train_model(model, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_losses = []
    val_losses = []
    
    no_improvement_epochs = 0
    max_no_improvement_epochs = 15
    
    for epoch in range(num_epochs):

        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("learning rate", param_group['lr'])

                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            metrics['mse'] = 0.0
            metrics['rmse'] = 0.0
            epoch_samples = 0

            
            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                
                for X, samples, masks, _ in dataloaders[phase]:

                    if phase == 'train':

                        X = X.to(device)
                        samples = samples.to(device)
                        masks = masks.to(device)

                    elif phase == 'val':
                        X = X.to(device)
                        samples = samples.to(device)
                        masks = masks.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                            outputs1 = model(X)
                            
                            loss = calc_loss_sparse(outputs1, samples, masks, metrics)

                            if phase == 'train':

                                loss.backward()
                                optimizer.step()

                    epoch_samples += X.size(0)
                    pbar.update(1)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['rmse']/epoch_samples
                        
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improvement_epochs = 0
            elif phase == 'val' and epoch_loss >= best_loss:
                no_improvement_epochs += 1
            
            train_losses.append(epoch_loss) if phase == 'train' else val_losses.append(epoch_loss)
        
        scheduler.step(epoch_loss)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        torch.cuda.empty_cache()

        if no_improvement_epochs >= max_no_improvement_epochs:
            print(f"No improvement in the last {max_no_improvement_epochs} epochs. Stopping training.")
            break
    
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_losses,val_losses,best_loss

#%% finetuning
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

print(config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=5, verbose=True)

model,train_losses,val_losses,best_loss = train_model(model, optimizer_ft, exp_lr_scheduler, config['num_epochs'])

best_val_loss = min(val_losses)

#%% save finetuned model
import os

file_path = f"finetuned_models/Finetuned_Model_{best_val_loss:.6f}.pt"
torch.save(model.state_dict(), file_path)
print(f"Finetuned model is saved at {file_path}")

#%% test after finetune
test_IDs = np.arange(0, len(Input_list_file_names), 1, dtype=int)

test_generator = finetune_dataloader.DataGenerator(
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

def test_loss(model):

    solution = pd.DataFrame(columns=["ID", "PL (dB)"])
    model.eval()
    index = -1
    for inputs, _, _, HW in DataLoader(test_generator, batch_size=1, shuffle=False, generator=torch.Generator(device = 'cuda')):
        index = index + 1

        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.squeeze().cpu().numpy()
            width = int(HW[0,1].item())
            height = int(HW[0,0].item())
            outputs1 = cv2.resize(outputs, (width, height), cv2.INTER_CUBIC)
            
            # Convert predicted values to db
            outputs1 = outputs1 * 255
            outputs1 = np.clip(outputs1, 13, 160)
            
            name = Input_list_file_names[index]

            # save as grayscale images
            cv2.imwrite(os.path.join(save_path, name + ".png"), outputs1)
            print(f"{name}.png has been saved.")

            # save as pandas DataFrame
            y_flat = np.expand_dims( outputs1.flatten() , 1)
            y_names = np.expand_dims( np.core.defchararray.add(  name + "_",  np.linspace(0, outputs1.size-1, outputs1.size).astype(int).astype(str) )  , 1)             
            y_data = np.concatenate( (y_names, y_flat ), axis = 1)
            y_pd = pd.DataFrame(data=y_data,index = np.linspace(0, outputs1.size-1, outputs1.size).astype(int) , columns= ["ID" ,"PL (dB)" ]) 
            solution = pd.concat([solution, y_pd], ignore_index=True)

    return solution

solution = test_loss(model)
solution.to_csv(f'Solution_check_task{task}_{best_val_loss:.6f}_finetuned.csv', index=False)

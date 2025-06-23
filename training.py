#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils, datasets, models
import dataloder.data_loaders_npy_deltaN_samples as data_loaders
import dataloder.data_loaders_npy1 as data_loaders_augment1
import dataloder.data_loaders_npy2 as data_loaders_augment2
import dataloder.data_loaders_npy3 as data_loaders_augment3
import warnings
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torch.optim import lr_scheduler
import torch.optim as optim
from torchsummary import summary
from lib import modules256Net_f_delta2, modules256Net_f_deltaN_samples, modules256Net_f_deltaN
from lib.SIP2Net import SIP2Net
import gc

seed_n = 42
torch.backends.cudnn.benchmark = True

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

def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss_mse = criterion(pred, target)
    loss_rmse = torch.sqrt(loss_mse)
    metrics['mse'] += loss_mse.item() * target.size(0)
    metrics['rmse'] += loss_rmse.item() * target.size(0)
    return loss_rmse

def calc_loss_mix(pred, target, metrics):
    criterion = nn.MSELoss()
    loss_mse = criterion(pred, target)
    ssim = SSIM(data_range=1.0).to(device)
    loss_ssim = 1 - ssim(pred, target)
    loss_rmse = torch.sqrt(loss_mse)
    loss = 0.8 * loss_rmse + 0.2 * loss_ssim

    metrics['mse'] += loss_mse.item() * target.size(0)
    metrics['rmse'] += loss_rmse.item() * target.size(0)
    metrics['1-ssim loss'] += loss_ssim.item() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase, log_file=None):
    outputs = ""
    for k in metrics.keys():
        outputs += "{}: {:.6f}, ".format(k, metrics[k] / epoch_samples)
    outputs = outputs.rstrip(", ")
    log_message = "{}: {}".format(phase, outputs)
    print(log_message)
    if log_file:
        log_file.write(log_message + "\n")
        log_file.flush()

def train_model(model, optimizer, scheduler, num_epochs, dataloaders, log_file=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_losses = []
    val_losses = []
    
    no_improvement_epochs = 0
    max_no_improvement_epochs = 15
    
    for epoch in range(num_epochs):
        epoch_message = 'Epoch {}/{}'.format(epoch+1, num_epochs)
        print(epoch_message)
        print('-' * 10)
        if log_file:
            log_file.write(epoch_message + "\n")
            log_file.write('-' * 10 + "\n")

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    lr_message = f"learning rate {param_group['lr']}"
                    print(lr_message)
                    if log_file:
                        log_file.write(lr_message + "\n")

                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            metrics['mse'] = 0.0
            metrics['rmse'] = 0.0
            metrics['1-ssim loss'] = 0.0
            epoch_samples = 0

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for X, y in dataloaders[phase]:
                    if phase == 'train' or phase == 'val':
                        X = X.to(device)
                        y = y.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs1 = model(X)
                        
                        loss = calc_loss_dense(outputs1, y, metrics)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    epoch_samples += X.size(0)
                    pbar.update(1)
            
            print_metrics(metrics, epoch_samples, phase, log_file)
            epoch_loss = metrics['rmse']/epoch_samples
            
            if phase == 'val' and epoch_loss < best_loss:
                save_message = "saving best model"
                print(save_message)
                if log_file:
                    log_file.write(save_message + "\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"trained_models/temp/Trained_Model.pt")
                no_improvement_epochs = 0
            elif phase == 'val' and epoch_loss >= best_loss:
                no_improvement_epochs += 1
            
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
        
        scheduler.step(epoch_loss)

        time_elapsed = time.time() - since
        time_message = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        print(time_message)
        if log_file:
            log_file.write(time_message + "\n")
        
        # torch.cuda.empty_cache()

        if no_improvement_epochs >= max_no_improvement_epochs:
            early_stop_message = f"No improvement in the last {max_no_improvement_epochs} epochs. Stopping training."
            print(early_stop_message)
            if log_file:
                log_file.write(early_stop_message + "\n")
            break
    
    best_loss_message = 'Best val loss: {:.6f}'.format(best_loss)
    print(best_loss_message)
    if log_file:
        log_file.write(best_loss_message + "\n")

    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss

def create_directories():
    dirs = ["trained_models/temp", "trained_models/task1", "trained_models/task2", "trained_models/task3", 
            "output/task1", "output/task2", "output/task3"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def main():
    config = {
        'task': 1,
        'model': 'delta',
        'resolution': 256,
        'n_channels': 5,
        'augment': False,
        'learning_rate': 0.001,
        'batch_size': 8,
        'num_epochs': 150,
        'train_shuffle': True,
        'pre_training': False,
        'pre_train_model': '0',
        'new_network': True,
    }

    create_directories()
    log_filepath = f"output\\task{config['task']}\\training_log.txt"
    log_file = open(log_filepath, "w", encoding="utf-8")
    
    log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Seed is {seed_n}\n")
    
    log_file.write(f"PyTorch version: {torch.__version__}\n")
    log_file.write(f"CUDA is available: {torch.cuda.is_available()}\n")
    
    log_file.write(f"Configuration: {config}\n")
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file.write(f"Using device: {device}\n")
    
    task = config['task']
    model_name = config['model']
    
    Input_list_file_names = []
    
    if task == 1:
        input_path = "E:/ICASSP/ICASSP2025_Dataset/Inputs/Task_1_ICASSP/"
        target_path = "E:/ICASSP/ICASSP2025_Dataset/Outputs/Task_1_ICASSP/"
        model_channel_path = "E:/ICASSP/ICASSP2025_Dataset/Inputs/" + "task" + str(task) + "_" + str(model_name) + "/"
        for b in range(1, 26): 
            for s in range(50):
                Input_list_file_names.append("B" + str(b) + "_Ant" + str(1) + "_f" + str(1) + "_S" + str(s))
    elif task == 2:
        input_path = "E:/ICASSP/ICASSP2025_Dataset/Inputs/Task_2_ICASSP/"
        target_path = "E:/ICASSP/ICASSP2025_Dataset/Outputs/Task_2_ICASSP/"
        model_channel_path = "E:/ICASSP/ICASSP2025_Dataset/Inputs/" + "task" + str(task) + "_" + str(model_name) + "/"
        for b in range(1, 26):
            for f in range(1, 4):
                for s in range(50):
                    Input_list_file_names.append("B" + str(b) + "_Ant" + str(1) + "_f" + str(f) + "_S" + str(s))
    elif task == 3:
        input_path = "E:/ICASSP/ICASSP2025_Dataset/Inputs/Task_3_ICASSP/"
        target_path = "E:/ICASSP/ICASSP2025_Dataset/Outputs/Task_3_ICASSP/"
        model_channel_path = "E:/ICASSP/ICASSP2025_Dataset/Inputs/" + "task" + str(task) + "_" + str(model_name) + "/"
        for b in range(1, 26):
            for a in range(1, 6):
                if a == 1:
                    for f in range(1, 4):
                        for s in range(50):
                            Input_list_file_names.append("B" + str(b) + "_Ant" + str(a) + "_f" + str(f) + "_S" + str(s))
                else:
                    for f in range(1, 4):
                        for s in range(80):
                            Input_list_file_names.append("B" + str(b) + "_Ant" + str(a) + "_f" + str(f) + "_S" + str(s))
    
    Input_list_IDs = np.arange(0, len(Input_list_file_names), 1, dtype=int)
    
    np.random.seed(42)
    np.random.shuffle(Input_list_IDs)
    train_split = int(0.9 * len(Input_list_IDs))
    val_split = int(1 * len(Input_list_IDs))
    
    train_IDs = Input_list_IDs[:train_split]
    val_IDs = Input_list_IDs[train_split:val_split]
    test_IDs = val_IDs
    
    log_file.write(f"Total samples: {len(Input_list_IDs)}\n")
    log_file.write(f"Training samples: {len(train_IDs)}\n")
    log_file.write(f"Validation samples: {len(val_IDs)}\n")
    # log_file.write(f"Test samples: {len(test_IDs)}\n")
    
    data_path=f"E:/ICASSP/ICASSP2025_Dataset/Data/Data_new8_delta/"
    dim_X=config['resolution']
    dim_y=config['resolution']
    n_channels=config['n_channels']
    train_generator = data_loaders.DataGenerator(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # device=torch.device("cpu"),
        list_IDs=train_IDs,
        file_names=Input_list_file_names,
        input_path=input_path,
        model_channel_path=model_channel_path,
        output_path=target_path,
        data_path=data_path,
        dim_X=dim_X,
        dim_y=dim_y,
        n_channels=n_channels,
        batch_size=1,
        shuffle=config['train_shuffle'],
        augment=False
    )
    
    val_generator = data_loaders.DataGenerator(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        list_IDs=val_IDs,
        file_names=Input_list_file_names,
        input_path=input_path,
        model_channel_path=model_channel_path,
        output_path=target_path,
        data_path=data_path,
        dim_X=dim_X,
        dim_y=dim_y,
        n_channels=n_channels,
        batch_size=1,
        shuffle=False,
        augment=False
    )
    
    test_generator = data_loaders.DataGenerator(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        list_IDs=test_IDs,
        file_names=Input_list_file_names,
        input_path=input_path,
        model_channel_path=model_channel_path,
        output_path=target_path,
        data_path=data_path,
        dim_X=dim_X,
        dim_y=dim_y,
        n_channels=n_channels,
        batch_size=1,
        shuffle=False,
        augment=False
    )
    
    if config['augment']:
        augment_generator1 = data_loaders_augment1.DataGenerator(
            list_IDs=train_IDs,
            file_names=Input_list_file_names,
            input_path=input_path,
            target_path=target_path,
            model_channel_path=model_channel_path,
            resolution=config['resolution']
        )
        
        augment_generator2 = data_loaders_augment2.DataGenerator(
            list_IDs=train_IDs,
            file_names=Input_list_file_names,
            input_path=input_path,
            target_path=target_path,
            model_channel_path=model_channel_path,
            resolution=config['resolution']
        )
        
        augment_generator3 = data_loaders_augment3.DataGenerator(
            list_IDs=train_IDs,
            file_names=Input_list_file_names,
            input_path=input_path,
            target_path=target_path,
            model_channel_path=model_channel_path,
            resolution=config['resolution']
        )
        
        train_dataset = ConcatDataset([train_generator, augment_generator1, augment_generator2, augment_generator3])
    else:
        train_dataset = train_generator
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['train_shuffle'], 
                            generator=torch.Generator(device='cuda')),
        'val': DataLoader(val_generator, batch_size=config['batch_size'], shuffle=False, 
                          generator=torch.Generator(device='cuda')),
        'test': DataLoader(test_generator, batch_size=config['batch_size'], shuffle=False, 
                           generator=torch.Generator(device='cuda'))
    }
    
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    model = modules256Net_f_deltaN_samples.RadioNet(inputs=6)
    model.cuda()
    
    if config['pre_training'] and config['new_network']:
        load_legacy_weights(model, f"trained_models/task{config['task']+2}/Trained_Model_{config['pre_train_model']}.pt")
    elif config['pre_training'] and not config['new_network']:
        model.load_state_dict(torch.load(f"trained_models/task{config['task']}/Trained_Model_{config['pre_train_model']}.pt"))
    
    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=config['learning_rate'], weight_decay=1e-2)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=5, verbose=True)
    
    log_file.write("Starting model training...\n")
    model, train_losses, val_losses, best_loss = train_model(
        model, optimizer_ft, exp_lr_scheduler, config['num_epochs'], dataloaders, log_file)
    
    best_val_loss = min(val_losses)
    output_filename = f"output/task{config['task']}/task{task}_{best_val_loss:.6f}.txt"
    file_path = f"trained_models/task{config['task']}/Trained_Model_{best_val_loss:.6f}.pt"
    
    if not os.path.exists(output_filename):
        with open(output_filename, 'w+') as f:
            f.write(f"Best val loss: {best_val_loss:.6f}\n")
        log_file.write(f"Training record saved to {output_filename}\n")
    
    if not os.path.exists(file_path):
        torch.save(model.state_dict(), file_path)
        log_file.write(f"Model saved to {file_path}\n")
    
    log_file.write(f"Model delta_1: {model.delta_1}\n")
    log_file.write(f"Model delta_2: {model.delta_2}\n")
    log_file.write(f"Model delta_3: {model.delta_3}\n")
    log_file.write(f"Model delta_4: {model.delta_4}\n")
    log_file.write(f"Model delta_6: {model.delta_6}\n")
    log_file.write(f"Model delta_10: {model.delta_10}\n")
    log_file.write(f"Model delta_23: {model.delta_23}\n")
    log_file.write(f"Model material_weights: {model.material_weights}\n")
    
    log_file.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()

    torch.cuda.empty_cache()
    for dataloader in dataloaders.values():
        del dataloader
    del model
    gc.collect()

if __name__ == "__main__":
    main()
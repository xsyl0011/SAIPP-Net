import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from skimage.io import imread
from utils.inputs_calculation import inputs_calculation

class DataGenerator(Dataset):
    """generates data for finetuning and test on rate0.02 (task 1 and task 2)"""

    def __init__(self, task, device, list_IDs, file_names, input_path, samples_path, sparse_mask_path, ground_truth_path,
                 dim_X, dim_y, n_channels = 4, 
                 batch_size=1, shuffle=False, augment=False):

        self.task = task
        self.device = device
        self.list_IDs = list_IDs
        self.file_names = file_names
        self.input_path = input_path
        self.samples_path = samples_path
        self.sparse_mask_path = sparse_mask_path
        self.ground_truth_path = ground_truth_path
        self.dim_X = dim_X # input dimensions
        self.dim_y = dim_y # output dimensions
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch and find IDs
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        filename = self.file_names[list_IDs_temp[0]]

        # Read Input images
        X, HW = self._generate_X(filename)
        X = X.astype(np.float32) / 255.0
        material_counts, model_channel = self._generate_material_counts_model_channel(filename, HW)
        samples, masks = self._generate_samples_masks(filename)

        f = np.full_like(X[2], 0.868)
        f = np.expand_dims(f, axis=0)
        X = np.concatenate((X, model_channel, f, material_counts), axis=0)
    
        X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        samples = torch.from_numpy(samples.astype(np.float32)/255.0).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)

        return X, samples, masks, HW

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def _generate_samples_masks(self, filename):
        if self.task == 1:
            # read already sampled GT for task 1
            samples = imread(self.samples_path + filename + ".png")
            resized_sampels = cv2.resize(samples, (self.dim_X, self.dim_y), interpolation=cv2.INTER_NEAREST )
            samples = np.expand_dims(resized_sampels, axis=0)
            mask = np.zeros_like(resized_sampels)
            mask[resized_sampels > 0] = 1
            mask = np.expand_dims(mask, axis=0)
        
        elif self.task == 2:
            # read and resize the sparse masks for rate0.5
            sparse_mask = imread(self.sparse_mask_path + filename + ".png")
            sparse_mask[sparse_mask == 255] = 1
            sparse_mask = cv2.resize(sparse_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # read and resize the ground truth
            ground_truth = imread(self.ground_truth_path + filename + ".png")
            ground_truth = cv2.resize(ground_truth, (256, 256), interpolation=cv2.INTER_NEAREST)
            sparse_measurement = ground_truth * sparse_mask
            resized_sampels = cv2.resize(sparse_measurement, (256, 256), interpolation=cv2.INTER_NEAREST)
            samples = np.expand_dims(resized_sampels, axis=0)

            mask = np.zeros_like(resized_sampels)
            mask[resized_sampels > 0] = 1
            mask = np.expand_dims(mask, axis=0)

        return samples, mask
    
    def _generate_material_counts_model_channel(self, filename, HW):
        material_maps, model_channel, _ = inputs_calculation(filename, HW)
        model_channel = cv2.resize(model_channel.astype(np.float32), (self.dim_X, self.dim_y), interpolation=cv2.INTER_CUBIC )
        model_channel = np.expand_dims(model_channel, axis=0)
        # LOS = cv2.resize(LOS.astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC )
        # LOS = np.expand_dims(LOS, axis=0)
        return material_maps.astype(np.float32), model_channel.astype(np.float32)#, LOS.astype(np.float32)

    def _generate_X(self, filename):

        X = imread(self.input_path + filename + ".png")
        HW = np.zeros(2)
        HW[0] = X.shape[0]
        HW[1] = X.shape[1]
        X = cv2.resize(X, (self.dim_X, self.dim_y), interpolation=cv2.INTER_CUBIC )
        X = np.moveaxis(X, -1, 0)
        X[:2] = X[:2] * 10
         
        return X.astype(np.float32), HW
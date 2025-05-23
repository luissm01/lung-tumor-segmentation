#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# In this notebook we will create the dataset class used to feed slices and corresponding segmentation masks to the network during training.
# It is identical to the CardiacDataset

# ## Imports
# 
# * pathlib for easy path handling
# * torch for dataset creation
# * numpy for file loading and processing
# * imgaug for data augmentation
# * matplotlib for demo

# In[28]:


from pathlib import Path

import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt


# ## DataSet Creation
# We need to implement the following functionality:
# 1. Create a list of all 2D slices. To so we need to extract all slices from all subjects
# 2. Extract the corresponding label path for each slice path
# 3. Load slice and label
# 4. Data Augmentation. Make sure that slice and mask are augmented identically. imgaug handles this for us, thus we will not use torchvision.transforms for that
# 5. Return slice and mask

# In[29]:


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params=None, tumor_oversampling_factor=20):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
        self.tumor_oversampling_factor = tumor_oversampling_factor
        self._precompute_weights()  # Nuevo: Precalcula pesos para oversampling

    @staticmethod
    def extract_files(root):
        """Extrae paths de todos los slices, ignorando slices vacíos si es necesario"""
        files = []
        for subject in root.glob("*"):
            slice_path = subject/"data"
            for slice_file in slice_path.glob("*"):
                files.append(slice_file)
        return files

    def _precompute_weights(self):
        """Precalcula pesos para oversampling de slices con tumor"""
        self.weights = []
        for file_path in self.all_files:
            mask_path = self.change_img_to_label_path(file_path)
            mask = np.load(mask_path)
            self.weights.append(self.tumor_oversampling_factor if np.any(mask) else 1)

    @staticmethod
    def change_img_to_label_path(path):
        parts = list(path.parts)
        parts[parts.index("data")] = "masks"
        return Path(*parts)

    def augment(self, slice, mask):
        """Aumentación con semilla aleatoria y transformaciones suaves"""
        random_seed = torch.randint(0, 1000000, (1,)).item()
        np.random.seed(random_seed)  # ✅ En lugar de imgaug.seed()
        
        # Configuración para preservar tumores pequeños
        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augment_params(
            image=slice,
            segmentation_maps=mask
        )
        return slice_aug, mask_aug.get_arr()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        
        # Carga con verificación de tipos
        slice = np.load(file_path).astype(np.float32)  # Aseguramos float32        
        mask = np.load(mask_path).astype(np.uint8)     # Máscara como entero

        if self.augment_params:
            slice, mask = self.augment(slice, mask)
            slice = np.clip(slice, 0, 1)

        # Expande dimensiones para el modelo (C, H, W)
        return (
            np.expand_dims(slice, 0), 
            np.expand_dims(mask, 0))


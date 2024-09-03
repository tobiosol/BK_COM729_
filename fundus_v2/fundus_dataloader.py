import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from fundus_v2 import fundus_img_dataset
from torch.utils.data import DataLoader, RandomSampler
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


class FundusDataLoader:
    def __init__(self, image_dir, csv_file, batch_size=16, shuffle=True, transform=None, train=True, num_augmentations=50):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.train = train
        self.num_augmentations = num_augmentations
        self.dataloader = self._create_dataloader()

    def get_loader(self):
        return self.dataloader

    def __iter__(self):
        return iter(self.dataloader)

    
    def _create_dataloader(self):
        dataset = fundus_img_dataset.FundusImageDataset(
            image_dir=self.image_dir,
            csv_file=self.csv_file,
            transform=self.transform,
            train=self.train,
            num_augmentations=self.num_augmentations
        )
        
        if self.train:
            # Extract features and labels
            features, labels = [], []
            for img, label in dataset:
                features.append(img.cpu().numpy().flatten())  # Move to CPU before converting to NumPy
                labels.append(label.cpu().numpy())  # Move to CPU before converting to NumPy
            
            features = np.array(features)
            labels = np.array(labels)
            
            # Apply SMOTE
            smote = SMOTE()
            features_resampled, labels_resampled = smote.fit_resample(features, labels)
            
            # Convert back to PyTorch tensors
            features_resampled = torch.tensor(features_resampled).view(-1, *dataset[0][0].shape)
            labels_resampled = torch.tensor(labels_resampled)
            
            # Create a new dataset with the resampled data
            dataset = TensorDataset(features_resampled, labels_resampled)
        
        # pin_memory = torch.cuda.is_available()
        
        if self.shuffle:
            sampler = RandomSampler(dataset)
            return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        
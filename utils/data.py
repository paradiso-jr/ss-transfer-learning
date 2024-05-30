import os
import torch
import numpy as np

from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class SleepEDFDataset(Dataset):
    """Dataset for sleep EDF 2018

    Args:
        dset_dir(str)
            path of sleep-edf npz data.
        augment(callable, optional)
            Optional data augmentation.
        transform(callable, optional)
            Optional transform to be applied on a sample.
    """
    def __init__(self, 
                dset_dir, 
                augment=None,
                transform=None, ):

        data = np.load(dset_dir)

        self.time_series = data['x']
        self.labels = data['y']
    
        self.augment = augment
        self.transform = transform
    def __len__(self):
        """ Return the number of time series samples."""
        return len(self.time_series)

    def __getitem__(self, index):
        """Return one sample of time series and its corresponding labels."""
        time_series = self.time_series[index]
        labels = self.labels[index]

        # set augmentation for contrastive learning
        # return the time series, augmented time series, and labels
        if self.augment:
            time_series_aug = self.augment(np.copy(time_series))
            
            if self.transform:
                time_series = self.transform(time_series)
        
            return (time_series, time_series_aug), labels

        if self.transform:
            time_series = self.transform(time_series)
        
        return time_series, labels
        

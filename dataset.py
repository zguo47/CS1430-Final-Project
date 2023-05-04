import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform

class Dataset1(Dataset):
    def __init__(self, gray_dir, color_dir, transform):
        # gray_dir: directory of grayscale images
        # color_dir: directory of color images
        # transform: the possible transformation
        # out_size: size of output image
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.transform = transform
        self.gray_filelist = os.listdir(self.gray_dir)
        self.color_filelist = os.listdir(self.color_dir)
    
    def __len__(self):
        return len(self.gray_filelist)
    
    def __getitem__(self, idx):
        '''
        Takes in an index and returns both its corresponding grayscale and color images
        '''
        gray_image = Image.open(self.gray_dir+self.gray_filelist[idx]).convert("L")
        color_image = Image.open(self.color_dir+self.color_filelist[idx]).convert("RGB")
        
        return self.transform(gray_image), self.transform(color_image)
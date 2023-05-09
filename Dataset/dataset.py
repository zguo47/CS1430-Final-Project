import torch
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skimage

class Dataset1(Dataset):
    def __init__(self, gray_dir, color_dir, transform, testing=False):
        # gray_dir: directory of grayscale images
        # color_dir: directory of color images
        # transform: the possible transformation
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.transform = transform
        self.gray_filelist = os.listdir(self.gray_dir)
        self.color_filelist = os.listdir(self.color_dir)
        self.testing = testing
    
    def __len__(self):
        return len(self.gray_filelist)
    
    def __getitem__(self, idx):
        '''
        Takes in an index and returns both its corresponding grayscale and color images
        '''
        if self.testing:
            idx = idx + 5000
            index = str(idx)
        else:
            if len(str(idx)) == 1:
                index = '000' + str(idx)
            elif len(str(idx)) == 2:
                index = '00' + str(idx)
            elif len(str(idx)) == 3:
                index = '0' + str(idx)
            elif len(str(idx)) == 4:
                index = str(idx)
        gray_image = Image.open(self.gray_dir+'image' + index + '.jpg').convert("L")
        color_image = Image.open(self.color_dir+'image' + index + '.jpg').convert("RGB")
        
        return self.transform(gray_image), self.transform(color_image)

class Dataset3(Dataset):
    def __init__(self, gray_dir, color_dir, transform):
        # gray_dir: directory of grayscale images
        # color_dir: directory of color images
        # transform: the possible transformation
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.transform = transform
        self.filelist = self.make_filelist(self.gray_dir, self.color_dir)
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        gray_filename = self.gray_dir + self.filelist[idx]
        color_filename = self.color_dir + self.filelist[idx]
        gray_image = Image.open(gray_filename).convert("L")
        color_image = Image.open(color_filename).convert("RGB")
        return self.transform(gray_image), self.transform(color_image)
    
    def make_filelist(self, gray_dir, color_dir):
        filelist1 = []
        for root, dirs, files in os.walk(gray_dir):
            for file in files:
                if file.endswith(".jpg"):
                    s = os.path.join(root, file)
                    if s.startswith(gray_dir):
                        filelist1.append(s.replace(gray_dir, ''))
        filelist2 = []
        for root, dirs, files in os.walk(color_dir):
            for file in files:
                if file.endswith(".jpg"):
                    s = os.path.join(root, file)
                    if s.startswith(color_dir):
                        filelist2.append(s.replace(color_dir, ''))
        filelist = []
        for name in filelist1:
            if name in filelist2:
                filelist.append(name)
        return filelist

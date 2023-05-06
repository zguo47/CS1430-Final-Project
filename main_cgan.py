import torch
import torch.nn as nn
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from models import *
from dataset import *
from functions_cgan import *


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CGAN().to(device)

    train_gray_dir = (
        "/content/drive/Shareddrives/CVFinalProject/data/dataset1/train_black/"
    )
    train_color_dir = (
        "/content/drive/Shareddrives/CVFinalProject/data/dataset1/train_color/"
    )
    test_gray_dir = (
        "/content/drive/Shareddrives/CVFinalProject/data/dataset1/test_black/"
    )
    test_color_dir = (
        "/content/drive/Shareddrives/CVFinalProject/data/dataset1/test_color/"
    )

    output_size = (224, 224)
    batch_size = 16

    transform = transforms.Compose(
        [transforms.Resize(output_size), transforms.ToTensor()]
    )

    train_dataset = Dataset1(train_gray_dir, train_color_dir, transform)
    test_dataset = Dataset1(test_gray_dir, test_color_dir, transform)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    num_epochs = 20
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    for epoch in range(num_epochs):
        train_losses = training_loop(model, optimizer, train_dataloader, device)
        test_losses = evaluation_loop(model, test_dataloader, device)
        epoch_train_loss = np.mean(train_losses)
        epoch_test_loss = np.mean(test_losses)
        print(
            f"Epoch {epoch+1}, avg train loss {epoch_train_loss:.3f}, avg test loss {epoch_test_loss:.3f}"
        )

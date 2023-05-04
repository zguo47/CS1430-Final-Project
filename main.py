from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np

from model import Colorization
from dataset import Dataset1
from functions import training_loop, evaluation_loop

def main():


    col = Colorization(256)

    val_number_of_images = 10
    batch_size = 100
    learning_rate = 0.001
    train_steps = 100
    num_epochs = 20
    optimizer =  torch.optim.Adam(col.parameters(), lr=learning_rate)

    train_gray_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/train_black/"
    train_color_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/train_color/"
    test_gray_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/test_black/"
    test_color_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/test_color/"

    output_size = (224, 224)
    batch_size = 16

    transform = transforms.Compose(
        [
            transforms.Resize(output_size),
            transforms.ToTensor()
        ]
    )

    train_dataset = Dataset1(train_gray_dir, train_color_dir, transform)
    test_dataset = Dataset1(test_gray_dir, test_color_dir, transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for epoch in range(num_epochs):
        train_losses = training_loop(col, train_steps, optimizer, train_dataloader, learning_rate)
        test_losses = evaluation_loop(col, test_dataloader)
        epoch_train_loss = np.mean(train_losses)
        epoch_test_loss = np.mean(test_losses)
        print(f"Epoch {epoch+1}, avg train loss {epoch_train_loss:.3f}, avg test loss {epoch_test_loss:.3f}")
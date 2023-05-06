from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms, utils
import numpy as np

from model import *
from dataset import *
from functions import *

def main():
    # Models and Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    col = Colorization(256).to(device)
    vgg_model = Vgg().to(device)
    # unet = Unet().to(device)
    # vgg_model = None
    learning_rate = 5e-4
    optimizer =  torch.optim.Adam(col.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    output_size = (224, 224)
    batch_size = 16
    num_epochs = 50

    # Prepare dataset1
    # On google drive
    # train_gray_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/train_black/"
    # train_color_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/train_color/"
    # test_gray_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/test_black/"
    # test_color_dir = "/content/drive/Shareddrives/CVFinalProject/data/dataset1/test_color/"
    # On computing cluster
    train_gray_dir = "/users/lliu58/scratch/data/train_black/"
    train_color_dir = "/users/lliu58/scratch/data/train_color/"
    test_gray_dir = "/users/lliu58/scratch/data/test_black/"
    test_color_dir = "/users/lliu58/scratch/data/test_color/"
    # On grid
    # train_gray_dir = "/home/lliu58/data/train_black/"
    # train_color_dir = "/home/lliu58/data/train_color/"
    # test_gray_dir = "/home/lliu58/data/test_black/"
    # test_color_dir = "/home/lliu58/data/test_color/"
    
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

    # Prepare dataset2
    # gray_dir = "/users/lliu58/scratch/dataset2/l/gray_scale.npy"
    # R_dir = "/users/lliu58/scratch/dataset2/ab/ab1.npy"
    # G_dir = "/users/lliu58/scratch/dataset2/ab/ab2.npy"
    # B_dir = "/users/lliu58/scratch/dataset2/ab/ab3.npy"
    # gray_dataset2, R_dataset2, G_dataset2, B_dataset2 = get_dataset2(gray_dir, R_dir, G_dir, B_dir)

    # Depending on where we execute code
    # save_location = "/home/lliu58/CS1430-Final-Project/experiment1/"
    # save_location = "/users/lliu58/data/lliu58/cv_final/experiment2/" 
    save_location = "/users/lliu58/data/lliu58/cv_final/experiment1/"

    best_testing_loss = 100000 # will be replaced by smaller values while training.
    batch_train_losses = []
    batch_test_losses = []
    for epoch in range(num_epochs):
        train_losses = training_loop(col, optimizer, loss_function, train_dataloader, device, vgg_model)
        test_losses = evaluation_loop(col, loss_function, test_dataloader, device, vgg_model)
        # train_losses = training_loop(unet, optimizer, loss_function, train_dataloader, device, vgg_model)
        # test_losses = evaluation_loop(unet, loss_function, test_dataloader, device, vgg_model)
        epoch_train_loss = np.mean(train_losses)
        epoch_test_loss = np.mean(test_losses)
        batch_train_losses.extend(train_losses)
        batch_test_losses.extend(test_losses)
        print(f"Epoch {epoch+1}, avg train loss {epoch_train_loss:.3f}, avg test loss {epoch_test_loss:.3f}")
        if best_testing_loss > epoch_test_loss:
            torch.save(col.state_dict(), save_location + "best_validation_model.pth")
            print(f"Epoch {epoch+1} is saved!")
            best_testing_loss = epoch_test_loss
    np.save(save_location+"training_losses.npy", np.array(batch_train_losses))
    np.save(save_location+"testing_losses.npy", np.array(batch_test_losses))

if __name__ == "__main__":
    main()

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms, utils
import numpy as np

from Model.model import *
from Dataset.dataset import *
from Model.functions import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CGAN().to(device)
# model.net_G.load_state_dict(torch.load("<path to model best training weights>"))
# train_gray_dir = "<path to training gray-scale data>"
# train_color_dir = "<path to training ground-truth RGB data>"
test_gray_dir = "<path to testing gray-scale data>"
test_color_dir = "<path to testing ground-truth RGB data>"
save_location = "<path to save location>"

output_size = (224, 224)
batch_size = 16

transform = transforms.Compose(
        [transforms.Resize(output_size), transforms.ToTensor()]
    )

# train_dataset = Dataset1(train_gray_dir, train_color_dir, transform)
# test_dataset = Dataset1(test_gray_dir, test_color_dir, transform, testing=True)
# train_dataloader = DataLoader(
#         dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
#     )
# test_dataloader = DataLoader(
#         dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
#     )

dataset3_gray_dir = "<path to dataset 3 training gray-scale data>"
dataset3_color_dir = "<path to dataset 3 training RGB data>"

train_dataset = Dataset3(dataset3_gray_dir, dataset3_color_dir, transform)
test_dataset = Dataset1(test_gray_dir, test_color_dir, transform, testing=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

num_epochs = 100
learning_rate = 5e-4
optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
best_testing_loss = 100000
batch_train_losses = []
batch_test_losses = []

for epoch in range(num_epochs):
    train_losses = training_loop_cgan(model, optimizer, train_dataloader, device)
    test_losses = evaluation_loop_cgan(model, test_dataloader, device)
    epoch_train_loss = np.mean(train_losses)
    epoch_test_loss = np.mean(test_losses)
    batch_train_losses.extend(train_losses)
    batch_test_losses.extend(test_losses)
    print(
            f"Epoch {epoch+1}, avg train loss {epoch_train_loss:.3f}, avg test loss {epoch_test_loss:.3f}"
        )
    # print(
    #         f"Epoch {epoch+1}, avg train loss {epoch_train_loss:.3f}"
    #     )
    if best_testing_loss > epoch_test_loss:
        torch.save(model.state_dict(), save_location + "best_model.pth")
        print(f"Epoch {epoch+1} is saved!")
        best_testing_loss = epoch_test_loss
np.save(save_location+"training_losses.npy", np.array(batch_train_losses))
np.save(save_location+"testing_losses.npy", np.array(batch_test_losses))
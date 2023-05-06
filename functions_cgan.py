import torch
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def training_loop(model, optimizer, train_dataloader, device):
    loss_list = []
    model.train()

    for batch_idx, img in enumerate(train_dataloader):
        imgs_l, imgs_true_ab = img
        imgs_l = imgs_l.to(device)
        imgs_true_ab = imgs_true_ab.to(device)
        optimizer.zero_grad()

        fake_prob, true_prob = model.forward(imgs_l, imgs_true_ab)
        batch_loss = model.loss(fake_prob, true_prob)
        if batch_idx % 10 == 0:
            print(batch_loss)
        batch_loss.backward()
        optimizer.step()
        loss_list.append(batch_loss.detach().cpu().numpy())

    return loss_list


def evaluation_loop(model, dataloader, device):
    loss_list = []
    model.eval()
    for batch_idx, img in enumerate(dataloader):
        imgs_l_val, imgs_true_ab_val = img
        imgs_l_val = imgs_l_val.to(device)
        imgs_true_ab_val = imgs_true_ab_val.to(device)

        fake_prob, true_prob = model.forward(imgs_l_val, imgs_true_ab_val)
        batch_loss = model.loss(fake_prob, true_prob)
        batch_loss.backward()
        loss_list.append(batch_loss.detach().cpu().numpy())
    return loss_list

"""
Given the L channel of an Lab image (range [-1, +1]), output a prediction over
the a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and fuse them with the conv output.
"""

import cv2
from torch.hub import load_state_dict_from_url
import torchvision
import torch.nn as nn
import torch


class Colorization(nn.Module):
    def __init__(self, depth_after_fusion):
        super().__init__()
        self.encoder = _build_encoder()
        self.fusion = FusionLayer()
        # self.dense = _build_dense()
        self.after_fusion = nn.Conv2d(1000+depth_after_fusion, depth_after_fusion, kernel_size = 1)
        # self.after_fusion = Conv2D(depth_after_fusion, (1, 1), activation="relu")
        self.decoder = _build_decoder(depth_after_fusion)

    def build(self, img_l):
        img_enc = self.encoder(img_l)
        vgg = Vgg()
        img_ab = torch.cat((img_l,img_l,img_l),1)
        img_emb = vgg(img_ab)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        return self.decoder(fusion)

def _build_encoder():
    model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    return model


def _build_decoder(encoding_depth):
    model = nn.Sequential(
            nn.Conv2d(encoding_depth, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    return model

def Vgg():
    model = torchvision.models.vgg16(weights="DEFAULT")
    for param in model.parameters():
                  param.requires_grad = False

    return model

class FusionLayer(nn.Module):
    def forward(self, inputs, mask=None):
        #check !!!
        imgs, embs = inputs # [16,256,28,28], [16,1000]
        (b,c,h,w) = imgs.shape # (batch_size,256,28,28)
        l = embs.shape[1] 
        embs = embs.unsqueeze(-1).unsqueeze(-1)
        embs = embs.expand(b, l, h, w)
        output = torch.cat ((imgs,embs),1)
        return output

    def compute_output_shape(self, input_shapes):
        # Must have 2 tensors as input
        assert input_shapes and len(input_shapes) == 2
        imgs_shape, embs_shape = input_shapes

        # The batch size of the two tensors must match
        assert imgs_shape[0] == embs_shape[0]

        # (batch_size, width, height, embedding_len + depth)
        return imgs_shape[:3] + (imgs_shape[3] + embs_shape[1],)
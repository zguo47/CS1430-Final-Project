import cv2
import torchvision
import torch
from torch import nn

#########################################################################################################################
# Deep Koalarization, but changed its pretrained weights 
# Given a grayscale image (range [0, 1]), output a prediction over its RGB channels in the range [0, 1].
# In the neck of the conv-deconv network use the features from a feature extractor and fuse them with the conv output.
#########################################################################################################################
class Colorization(nn.Module):
    def __init__(self, depth_after_fusion):
        super().__init__()
        self.encoder = _build_encoder()
        self.fusion = FusionLayer()
        self.after_fusion = nn.Conv2d(1000+depth_after_fusion, depth_after_fusion, kernel_size = 1)
        self.decoder = _build_decoder(depth_after_fusion)

    def forward(self, img_l, vgg):
        img_enc = self.encoder(img_l)
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
    model = torchvision.models.vgg16(weights="DEFAULT") # must use this in order to run on Department machines.
    # model = torchvision.models.vgg16(pretrained=True) # must use this in order to run on Department machines.
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

#########################################################################################################################
# Unet itself could be trained to colorize images
#########################################################################################################################
class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers
        self.encoder0 = nn.Sequential(
                            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same'), 
                            nn.BatchNorm2d(64), 
                            nn.LeakyReLU(0.2), 
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(64), 
                            nn.LeakyReLU(0.2)
                        )
        self.encoder1 = nn.Sequential(
                            nn.MaxPool2d(kernel_size=2, stride=2), 
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'), 
                            nn.BatchNorm2d(128), 
                            nn.LeakyReLU(0.2), 
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(128), 
                            nn.LeakyReLU(0.2)
                        )
        self.encoder2 = nn.Sequential(
                            nn.MaxPool2d(kernel_size=2, stride=2), 
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'), 
                            nn.BatchNorm2d(256), 
                            nn.LeakyReLU(0.2), 
                            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(256), 
                            nn.LeakyReLU(0.2)
                        )
        self.encoder3 = nn.Sequential(
                            nn.MaxPool2d(kernel_size=2, stride=2), 
                            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'), 
                            nn.BatchNorm2d(512), 
                            nn.LeakyReLU(0.2), 
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(512), 
                            nn.LeakyReLU(0.2)
                        )
        self.encoder4 = nn.Sequential(
                            nn.MaxPool2d(kernel_size=2, stride=2), 
                            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'), 
                            nn.BatchNorm2d(1024), 
                            nn.LeakyReLU(0.2), 
                            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                            nn.BatchNorm2d(1024), 
                            nn.LeakyReLU(0.2)
                        )
        self.up_sample4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
                        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding='same'), 
                        nn.BatchNorm2d(512), 
                        nn.ReLU(0.2), 
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
                        nn.BatchNorm2d(512), 
                        nn.ReLU(0.2)
                    )
        self.up_sample3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding='same'), 
                        nn.BatchNorm2d(256), 
                        nn.ReLU(0.2), 
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
                        nn.BatchNorm2d(256), 
                        nn.ReLU(0.2)
                    )
        self.up_sample2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding='same'), 
                        nn.BatchNorm2d(128), 
                        nn.ReLU(0.2), 
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
                        nn.BatchNorm2d(128), 
                        nn.ReLU(0.2)
                    )
        self.up_sample1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding='same'), 
                        nn.BatchNorm2d(64), 
                        nn.ReLU(0.2), 
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
                        nn.BatchNorm2d(64), 
                        nn.ReLU(0.2)
                    )
        self.conv0 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)
        
    
    def forward(self, x):
        # forward pass of the model
        # denote x as input grayscale image
        # returns y_pred as predicted color image
        x0 = self.encoder0(x) # shape [B, 64, H, W]
        x1 = self.encoder1(x0) # shape [B, 128, H/2, W/2]
        x2 = self.encoder2(x1) # shape [B, 256, H/4, W/4]
        x3 = self.encoder3(x2) # shape [B, 512, H/8, W/8]
        x4 = self.encoder4(x3) # shape [B, 1024, H/16, W/16]
        x4 = self.up_sample4(x4) # shape [B, 512, H/8, W/8]
        x3 = self.conv4(torch.cat((x3, x4), 1)) # shape [B, 512, H/8, W/8]
        x3 = self.up_sample3(x3) # shape [B, 256, H/4, W/4]
        x2 = self.conv3(torch.cat((x2, x3), 1)) # shape [B, 256, H/4, W/4]
        x2 = self.up_sample2(x2) # shape [B, 128, H/2, W/2]
        x1 = self.conv2(torch.cat((x1, x2), 1)) # shape [B, 128, H/2, W/2]
        x1 = self.up_sample1(x1) # shape [B, 64, H, W]
        x0 = self.conv1(torch.cat((x0, x1), 1)) # shape [B, 64, H, W]
        return self.conv0(x0)

#########################################################################################################################
# Conditional GAN, where the generator is the Unet defined above
#########################################################################################################################
class Dis(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers
        self.block1 = nn.Sequential(
                            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=(1,1)), 
                            nn.LeakyReLU(0.2, inplace=True) 
                        )
        self.block2 = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=(1,1), bias=False), 
                            nn.BatchNorm2d(128), 
                            nn.LeakyReLU(0.2, inplace=True) 
                        )
        self.block3 = nn.Sequential(
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=(1,1), bias=False), 
                            nn.BatchNorm2d(256), 
                            nn.LeakyReLU(0.2, inplace=True) 
                        )
        self.block4 = nn.Sequential(
                            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=(1,1), bias=False), 
                            nn.BatchNorm2d(512), 
                            nn.LeakyReLU(0.2, inplace=True) 
                        )
        self.block5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=(1,1))
        self.dense = nn.Sequential(nn.Flatten(), 
                                   nn.Linear(in_features=676, out_features=15),
                                   nn.Linear(in_features=15, out_features=1), 
                                   nn.Sigmoid())

    
    def forward(self, img_ab, img_l):
        img_comb = torch.cat((img_ab, img_l), dim=1)
        layers = [self.block1, self.block2, self.block3, self.block4, self.block5]
        for layer in layers:
          img_comb = layer(img_comb)
        img_comb = self.dense(img_comb)
        return img_comb

class CGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lda = 50 # hyperparameter for l2 loss
        self.buffer = 1e-2
        self.net_G = Unet()
        self.net_D = Dis()

    def forward(self, imgs_l, imgs_ab):
        fake_color = self.net_G(imgs_l)
        fake_prob = self.net_D(fake_color, imgs_l)
        true_prob = self.net_D(imgs_ab, imgs_l)
        return fake_prob, true_prob, fake_color

    def loss(self, fake_prob, true_prob, fake_color, imgs_ab):
        # fake_prob and true_prob are both of shape [B, 1]
        gan_loss = torch.mean(torch.log(true_prob + self.buffer) + torch.log(1 - fake_prob + self.buffer))
        l1loss = nn.functional.l1_loss(fake_color, imgs_ab)
        print(l1loss)
        return gan_loss + l1loss

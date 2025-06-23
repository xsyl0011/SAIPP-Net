import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def convrelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True), 
    )

maxpool = nn.MaxPool2d(2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

class SDUblock(nn.Module):
    def __init__(self, in_channels, n_out):
        super(SDUblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_out // 2, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(n_out // 2, n_out // 4, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(n_out // 4, n_out // 8, kernel_size=3, padding=6, dilation=6)
        self.conv4 = nn.Conv2d(n_out // 8, n_out // 16, kernel_size=3, padding=9, dilation=9)
        self.conv5 = nn.Conv2d(n_out // 16, n_out // 16, kernel_size=3, padding=12, dilation=12)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(out4))
        return torch.cat([out1, out2, out3, out4, out5], dim=1)

def convreluT(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU(inplace=True),
    )

def upsample(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
    )


class RadioNet(nn.Module):

    def __init__( self,inputs=6):
        
        super().__init__()
        self.inputs = inputs
        # self.z = z

        self.encode1_conv = convrelu(inputs-1, 32)
        self.encode1_sdu = SDUblock(32, 32)
        self.maxpool = maxpool

        self.encode2_conv = convrelu(32, 64)
        self.encode2_sdu = SDUblock(64, 64)

        self.encode3_conv = convrelu(64, 128)
        self.encode3_sdu = SDUblock(128, 128)

        self.encode4_conv = convrelu(128, 256)
        self.encode4_sdu = SDUblock(256, 256)

        self.encode5_conv = convrelu(256, 512)
        self.encode5_sdu = SDUblock(512, 512)

        self.bottleneck_conv = convrelu(512, 1024)
        self.bottleneck_sdu = SDUblock(1024, 1024)
        
        self.upconv1 = convreluT(1024,512)
        self.decode1_conv = convrelu(512*2, 512)
        self.decode1_sdu = SDUblock(512, 512)

        self.upconv2 = convreluT(512,256)
        self.decode2_conv = convrelu(256*2, 256)
        self.decode2_sdu = SDUblock(256, 256)

        self.upconv3 = convreluT(256,128)
        self.decode3_conv = convrelu(128*2, 128)
        self.decode3_sdu = SDUblock(128, 128)

        self.upconv4 = convreluT(128,64)
        self.decode4_conv = convrelu(64*2, 64)
        self.decode4_sdu = SDUblock(64, 64)

        self.upconv5 = convreluT(64,32)
        self.decode5_conv = convrelu(32*2, 32)
        self.decode5_sdu = SDUblock(32, 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        
        self.delta_1 = nn.Parameter(torch.tensor(1.0))
        self.delta_2 = nn.Parameter(torch.tensor(1.0))
        self.delta_3 = nn.Parameter(torch.tensor(1.0))
        self.delta_4 = nn.Parameter(torch.tensor(1.0))
        self.delta_6 = nn.Parameter(torch.tensor(1.0))
        self.delta_10 = nn.Parameter(torch.tensor(1.0))
        self.delta_23 = nn.Parameter(torch.tensor(1.0))

        self.material_weights = nn.Parameter(torch.ones(7))

    def forward(self, input, z=None):

        model = input[:, 3, :, :]
        f = input[:, 4, :, :]

        materials = input[:, 5:12, :, :]
        material_contribution = torch.sum(materials * self.material_weights.view(1, 7, 1, 1), dim=1)
        
        model_channel = (model + material_contribution) / 255.0
        input0 = torch.cat([input[:, 0:3, :, :], model_channel.unsqueeze(1), f.unsqueeze(1)], dim=1)

        # encoder
        encode1_conv = self.encode1_conv(input0)
        encode1_sdu = self.encode1_sdu(encode1_conv)
        encode1_pool = self.maxpool(encode1_sdu)

        encode2_conv = self.encode2_conv(encode1_pool)
        encode2_sdu = self.encode2_sdu(encode2_conv)
        encode2_pool = self.maxpool(encode2_sdu)

        encode3_conv = self.encode3_conv(encode2_pool)
        encode3_sdu = self.encode3_sdu(encode3_conv)
        encode3_pool = self.maxpool(encode3_sdu)

        encode4_conv = self.encode4_conv(encode3_pool)
        encode4_sdu = self.encode4_sdu(encode4_conv)
        encode4_pool = self.maxpool(encode4_sdu)

        encode5_conv = self.encode5_conv(encode4_pool)
        encode5_sdu = self.encode5_sdu(encode5_conv)
        encode5_pool = self.maxpool(encode5_sdu)
        
        # bottleneck
        bottleneck_conv = self.bottleneck_conv(encode5_pool)
        bottleneck_sdu = self.bottleneck_sdu(bottleneck_conv)

        # decoder
        decode1_up = self.upconv1(bottleneck_sdu)
        decode1_cat = torch.cat([decode1_up,encode5_sdu ], dim=1)
        decode1_conv = self.decode1_conv(decode1_cat)
        decode1_sdu = self.decode1_sdu(decode1_conv)

        decode2_up = self.upconv2(decode1_sdu)
        decode2_cat = torch.cat([decode2_up,encode4_sdu ], dim=1)
        decode2_conv = self.decode2_conv(decode2_cat)
        decode2_sdu = self.decode2_sdu(decode2_conv)

        decode3_up = self.upconv3(decode2_sdu)
        decode3_cat = torch.cat([decode3_up,encode3_sdu ], dim=1)
        decode3_conv = self.decode3_conv(decode3_cat)
        decode3_sdu = self.decode3_sdu(decode3_conv)

        decode4_up = self.upconv4(decode3_sdu)
        decode4_cat = torch.cat([decode4_up,encode2_sdu ], dim=1)
        decode4_conv = self.decode4_conv(decode4_cat)
        decode4_sdu = self.decode4_sdu(decode4_conv)

        decode5_up = self.upconv5(decode4_sdu)
        decode5_cat = torch.cat([decode5_up,encode1_sdu ], dim=1)
        decode5_conv = self.decode5_conv(decode5_cat)
        decode5_sdu = self.decode5_sdu(decode5_conv)

        output = self.output(decode5_sdu)

        return output
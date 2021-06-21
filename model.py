import torch
import torch.nn as nn
import numpy as np


class Conv2DLayerBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=True):
        super(Conv2DLayerBN, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        if activation:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv2d(x)))


class Deconv2DLayerBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(Deconv2DLayerBN, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=strides)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.deconv2d(x)))


class UNet2D(nn.Module):
    def __init__(self, nchannels=1, nlabels=4):
        super(UNet2D, self).__init__()
        self.nchannels = nchannels
        self.nlabels = nlabels
        self.conv1_1 = Conv2DLayerBN(nchannels, 64)
        self.conv1_2 = Conv2DLayerBN(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = Conv2DLayerBN(64, 128)
        self.conv2_2 = Conv2DLayerBN(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = Conv2DLayerBN(128, 256)
        self.conv3_2 = Conv2DLayerBN(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4_1 = Conv2DLayerBN(256, 512)
        self.conv4_2 = Conv2DLayerBN(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5_1 = Conv2DLayerBN(512, 1024)
        self.conv5_2 = Conv2DLayerBN(1024, 1024)
        self.upconv4 = Deconv2DLayerBN(1024, nlabels)
        self.conv6_1 = Conv2DLayerBN(512 + nlabels, 512)
        self.conv6_2 = Conv2DLayerBN(512, 512)
        self.upconv3 = Deconv2DLayerBN(512, nlabels)
        self.conv7_1 = Conv2DLayerBN(256 + nlabels, 256)
        self.conv7_2 = Conv2DLayerBN(256, 256)
        self.upconv2 = Deconv2DLayerBN(256, nlabels)
        self.conv8_1 = Conv2DLayerBN(128 + nlabels, 128)
        self.conv8_2 = Conv2DLayerBN(128, 128)
        self.upconv1 = Deconv2DLayerBN(128, nlabels)
        self.conv9_1 = Conv2DLayerBN(64 + nlabels, 64)
        self.conv9_2 = Conv2DLayerBN(64, 64)
        self.pred = Conv2DLayerBN(64, 4, 1, False)

    def crop_and_concat_layer(self, inputs, axis):
        output_size = list(inputs[0].shape)
        concat_inputs = [inputs[0]]

        for ii in range(1, len(inputs)):
            larger_size = list(inputs[ii].shape)
            start_crop = np.subtract(larger_size, output_size) // 2
            cropped_tensor = inputs[ii][:, :, start_crop[2]:start_crop[2] + output_size[2],
                             start_crop[3]:start_crop[3] + output_size[3]]
            concat_inputs.append(cropped_tensor)
        return torch.cat(concat_inputs, dim=axis)

    def forward(self, x):
        images_padded = nn.functional.pad(x, (92, 92, 92, 92), 'constant', 0)

        conv1_1 = self.conv1_1(images_padded)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        upconv4 = self.upconv4(conv5_2)
        concat4 = self.crop_and_concat_layer([upconv4, conv4_2], axis=1)

        conv6_1 = self.conv6_1(concat4)
        conv6_2 = self.conv6_2(conv6_1)
        upconv3 = self.upconv3(conv6_2)
        concat3 = self.crop_and_concat_layer([upconv3, conv3_2], axis=1)

        conv7_1 = self.conv7_1(concat3)
        conv7_2 = self.conv7_2(conv7_1)
        upconv2 = self.upconv2(conv7_2)
        concat2 = self.crop_and_concat_layer([upconv2, conv2_2], axis=1)

        conv8_1 = self.conv8_1(concat2)
        conv8_2 = self.conv8_2(conv8_1)
        upconv1 = self.upconv1(conv8_2)
        concat1 = self.crop_and_concat_layer([upconv1, conv1_2], axis=1)

        conv9_1 = self.conv9_1(concat1)
        conv9_2 = self.conv9_2(conv9_1)

        pred = self.pred(conv9_2)
        return pred
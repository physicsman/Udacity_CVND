## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, in_channels):
        super(Net, self).__init__()
        
        assert isinstance(in_channels, int)
        self.in_channels = in_channels
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.output_len = 136
        
        self.alpha = 0.05
        self.dropout = 0.3

        self.filters = [64, 128, 256, 512] # [32, 64, 128, 256] # [16, 32, 64, 128]
        self.kernel_size =  [3, 2, 3, 2]# [3, 2, 3, 2]
        self.pool_size = [2, 2, 2, 2]
        
        self.conv1a = nn.Conv2d(self.in_channels, self.filters[0], self.kernel_size[0], padding=1)
        self.conv1a_bn = nn.BatchNorm2d(self.filters[0])
        # LeakyReLU
        # x = F.leaky_relu(self.conv1a_bn(self.conv1a(x)), self.alpha)
        self.conv1b = nn.Conv2d(self.filters[0], self.filters[0], self.kernel_size[0], padding=1)
        self.maxpool1 = nn.MaxPool2d(self.pool_size[0])
        self.conv1b_bn = nn.BatchNorm2d(self.filters[0])
        # LeakyReLU
        self.drop1 = nn.Dropout2d(self.dropout)
        # x = self.drop1(F.leaky_relu(self.conv1b_bn(self.maxpool1(self.conv1b(x))), self.alpha))
        
        self.conv2a = nn.Conv2d(self.filters[0], self.filters[1], self.kernel_size[1], padding=1)
        self.conv2a_bn = nn.BatchNorm2d(self.filters[1])
        # LeakyReLU
        # x = F.leaky_relu(self.conv2a_bn(self.conv2a(x)), self.alpha)
        self.conv2b = nn.Conv2d(self.filters[1], self.filters[1], self.kernel_size[1], padding=0)
        self.conv2b_bn = nn.BatchNorm2d(self.filters[1])
        # LeakyReLU
        # x = F.leaky_relu(self.conv2b_bn(self.conv2b(x)), self.alpha)
        self.conv2c = nn.Conv2d(self.filters[1], self.filters[1], self.kernel_size[1], padding=1)
        self.maxpool2 = nn.MaxPool2d(self.pool_size[1])
        self.conv2c_bn = nn.BatchNorm2d(self.filters[1])
        # LeakyReLU
        self.drop2 = nn.Dropout2d(self.dropout)
        # x = self.drop2(F.leaky_relu(self.conv2c_bn(self.maxpool2(self.conv2c(x))), self.alpha))

        self.conv3a = nn.Conv2d(self.filters[1], self.filters[2], self.kernel_size[2], padding=1)
        self.conv3a_bn = nn.BatchNorm2d(self.filters[2])
        # LeakyReLU
        # x = F.leaky_relu(self.conv3a_bn(self.conv3a(x)), self.alpha)
        self.conv3b = nn.Conv2d(self.filters[2], self.filters[2], self.kernel_size[2], padding=1)
        self.conv3b_bn = nn.BatchNorm2d(self.filters[2])
        # LeakyReLU
        # x = F.leaky_relu(self.conv3b_bn(self.conv3b(x)), self.alpha)
        self.conv3c = nn.Conv2d(self.filters[2], self.filters[2], self.kernel_size[2], padding=1)
        self.maxpool3 = nn.MaxPool2d(self.pool_size[2])
        self.conv3c_bn = nn.BatchNorm2d(self.filters[2])
        # LeakyReLU
        self.drop3 = nn.Dropout2d(self.dropout)
        # x = self.drop3(F.leaky_relu(self.conv3c_bn(self.maxpool3(self.conv3c(x))), self.alpha))
        
        self.conv4a = nn.Conv2d(self.filters[2], self.filters[3], self.kernel_size[3], padding=1)
        self.conv4a_bn = nn.BatchNorm2d(self.filters[3])
        # LeakyReLU
        # x = F.leaky_relu(self.conv4a_bn(self.conv4a(x)), self.alpha)
        self.conv4b = nn.Conv2d(self.filters[3], self.filters[3], self.kernel_size[3], padding=0)
        self.conv4b_bn = nn.BatchNorm2d(self.filters[3])
        # LeakyReLU
        # x = F.leaky_relu(self.conv4b_bn(self.conv4b(x)), self.alpha)
        self.conv4c = nn.Conv2d(self.filters[3], self.filters[3], self.kernel_size[3], padding=1)
        self.maxpool4 = nn.MaxPool2d(self.pool_size[3])
        self.conv4c_bn = nn.BatchNorm2d(self.filters[3])
        # LeakyReLU
        self.drop4 = nn.Dropout2d(self.dropout)
        # x = self.drop4(F.leaky_relu(self.conv4c_bn(self.maxpool4(self.conv4c(x))), self.alpha))
        
        self.conv5 = nn.Conv2d(self.filters[3], self.output_len, self.kernel_size[3])
        self.maxpool5 = nn.MaxPool2d(self.pool_size[3], self.pool_size[3])
        self.conv5_bn = nn.BatchNorm2d(self.output_len)
        # LeakyReLU
        # x = F.leaky_relu(self.conv5_bn(self.maxpool5(self.conv5(x))), self.alpha)
        
        # Global Average Pooling 
        self.gapool = nn.AdaptiveAvgPool2d(1)
        # x = F.adaptive_avg_pool2d(x,1)
        # x = x.view(x.size(0), -1)
        self.linear = nn.Linear(self.output_len, self.output_len)
        # Tanh
        # x = F.tanh(self.linear(x))
        
        
    def forward(self, x, n=None):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = F.leaky_relu(self.conv1a_bn(self.conv1a(x)), self.alpha)
        if n == 1: 
            return x
        x = self.drop1(F.leaky_relu(self.conv1b_bn(self.maxpool1(self.conv1b(x))), self.alpha))
        if n == 2: 
            return x
        x = F.leaky_relu(self.conv2a_bn(self.conv2a(x)), self.alpha)
        if n == 3: 
            return x
        x = F.leaky_relu(self.conv2b_bn(self.conv2b(x)), self.alpha)
        if n == 4: 
            return x
        x = self.drop2(F.leaky_relu(self.conv2c_bn(self.maxpool2(self.conv2c(x))), self.alpha))
        if n == 5: 
            return x
        x = F.leaky_relu(self.conv3a_bn(self.conv3a(x)), self.alpha)
        if n == 6: 
            return x
        x = F.leaky_relu(self.conv3b_bn(self.conv3b(x)), self.alpha)
        if n == 7: 
            return x
        x = self.drop3(F.leaky_relu(self.conv3c_bn(self.maxpool3(self.conv3c(x))), self.alpha))
        if n == 8: 
            return x
        x = F.leaky_relu(self.conv4a_bn(self.conv4a(x)), self.alpha)
        if n == 9: 
            return x
        x = F.leaky_relu(self.conv4b_bn(self.conv4b(x)), self.alpha)
        if n == 10: 
            return x
        x = self.drop4(F.leaky_relu(self.conv4c_bn(self.maxpool4(self.conv4c(x))), self.alpha))
        if n == 11: 
            return x
        x = F.leaky_relu(self.conv5_bn(self.maxpool5(self.conv5(x))), self.alpha)
        if n == 12: 
            return x
        
        #x = F.adaptive_avg_pool2d(x,1)
        x = self.gapool(x)
        if n == 13: 
            return x
        x = x.view(x.size(0), -1) # prep for linear layer
        x = F.tanh(self.linear(x))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

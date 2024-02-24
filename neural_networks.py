import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
    
class Extra_layer(nn.Module):
    def __init__(self):
        super(Extra_layer, self).__init__()
        self.bin = nn.Linear(10, 2)     
    
    def forward(self, x):
        x = self.bin(x)
        return F.log_softmax(x,dim=1)
    
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with leaky relu activation function
        # and maxpooling after
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with leaky relu activation function
        x = F.leaky_relu(self.t_conv1(x))
        # output layer
        x = self.t_conv2(x)
                
        return x
    
    def encode(self, x):
        ## encode ##
        # add hidden layers with leaky relu activation function
        # and maxpooling after
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        return x
    
    def decode(self, x):
        ## decode ##
        # add transpose conv layers, with leaky relu activation function
        x = F.leaky_relu(self.t_conv1(x))
        # output layer
        x = self.t_conv2(x)
        return x


class BinaryMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_digit=8):
        self.mnist = MNIST(root, train=train, transform=transform, download=True)
        self.target_digit = target_digit

    def __getitem__(self, index):
        img, target = self.mnist[index]
        # Modify the label to be 1 if it's the target digit, otherwise 0
        target = 1 if target == self.target_digit else 0
        return img, target

    def __len__(self):
        return len(self.mnist)
    
    
class ClassificationMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_digit=1, class_0_digit=0):
        self.mnist = MNIST(root, train=train, transform=transform, download=True)
        self.target_digit = target_digit
        self.class_0_digit = class_0_digit

        # Find indices of samples corresponding to the target digits
        self.indices = []
        for i, (_, target) in enumerate(self.mnist):
            if target == self.target_digit or target == self.class_0_digit:
                self.indices.append(i)

    def __getitem__(self, index):
        # Get the index of the sample within the selected indices
        index = self.indices[index]
        img, target = self.mnist[index]
        # Modify the label to be 1 if it's the target digit, otherwise 0
        if target == self.target_digit:
            label = 1
        elif target == self.class_0_digit:
            label = 0
        else:
            # This should never happen, but you can handle it if needed
            raise ValueError(f"Unexpected target value: {target}")
        return img, label

    def __len__(self):
        return len(self.indices)
    
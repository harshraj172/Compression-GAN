import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Teacher(nn.Module):
    """
    A Teacher model giving the latent represntation of 
    input using pretrained model.
    """
    def __init__(self, pretrained_model="vgg16"):
        super(Teacher, self).__init__()

        self.model = load_pretrained(pretrained_model)
        output_dim = self.model
    def load_pretrained(name):
        if name=="alexnet":
            model = models.alexnet(pretrained=True)
        elif name=="vgg16":
            model = models.vgg16(pretrained=True)
        elif name=="resnet18":
            model = models.resnet18(pretrained=True)
        return model
    
    def forward(self, x):
        x = self.model(x)
        return x


class Student(nn.Module):
    """
    A simple CNN module giving the latent
    a latent represntation as output.
    """
    def __init__(self, output_dim):
        super(Student).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        h = x
        x = self.fc_1(x)
        x = F.relu(x)

        x = self.fc_2(x)
        x = F.relu(x)

        x = self.fc_3(x)
        return x, h



class Discriminator(nn.Module):
    """
    Discriminator to give the probability
    of the latent representation to be real or fake.
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)

        return validity


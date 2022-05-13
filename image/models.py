import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Teacher(nn.Module):
    """
    A Teacher model giving the latent represntation of 
    input using pretrained model.
    """
    output_dim=1000
    def __init__(self, pretrained_model="resnet18"):
        super(Teacher, self).__init__()
        self.model = self.load_pretrained(pretrained_model)
        
    def load_pretrained(self, name):
        if name=="alexnet":
            model = models.alexnet(pretrained=True, progress=False)
        elif name=="vgg16":
            model = models.vgg16(pretrained=True, progress=False)
        elif name=="resnet18":
            model = models.resnet18(pretrained=True, progress=False)
        return model
    
    def forward(self, x):
        x = self.model(x)
        return x


class Student(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(Student, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
        )

    def forward(self, x):
        validity = self.model(x)

        return validity
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self, input_dim, num_classes).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


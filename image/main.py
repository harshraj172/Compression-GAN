import os
import wandb 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

from data import *
from models import Teacher, Student, Discriminator 
from train import Trainer

def main(train_dataroot,
        val_dataroot,
        imagesize,
        pretrained_model,
        lr,
        n_epochs,
        batch_size,
        device,
        save_dir,):
        
    # Configure data loader
    os.makedirs(train_dataroot, exist_ok=True)
    os.makedirs(val_dataroot, exist_ok=True)
    trainloader_T = torch.utils.data.DataLoader(CIFAR10(dataroot=train_dataroot,
                                                    transform=transforms.Compose([
                                                    transforms.Resize(imagesize),
                                                    transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    ])),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            )
    trainloader_S = torch.utils.data.DataLoader(CIFAR10(dataroot=train_dataroot,
                                                    transform=transforms.Compose([
                                                    transforms.Resize(imagesize),
                                                    transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    ])),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            )
    valloader_T = torch.utils.data.DataLoader(CIFAR10(dataroot=val_dataroot,
                                                    transform=transforms.Compose([
                                                    transforms.Resize(imagesize),
                                                    transforms.ToTensor()
                                                    ]), train=False),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            )
    valloader_S = torch.utils.data.DataLoader(CIFAR10(dataroot=val_dataroot,
                                                    transform=transforms.Compose([
                                                    transforms.Resize(imagesize),
                                                    transforms.ToTensor()
                                                    ]), train=False),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            )

    # Initialize Teacher, Student, Discriminator
    teacher = Teacher(pretrained_model).to(device)
    output_dim = teacher.output_dim
    student = Student(output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)

    # Loss function
    loss = torch.nn.BCELoss().to(device)

    # Optimizers
    optimizer_S = torch.optim.Adam(student.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # ----------
    #  Training
    # ----------
    config = {
      "dataset": "CIFAR10",
      "imagesize": imagesize,  
      "model": pretrained_model,
      "learning_rate": lr,
      "batch_size": batch_size,
    }

    wandb.init(project="Compression-GAN", entity="harsh1729", config=config)

    trainer = Trainer(
                      teacher, 
                      student,
                      discriminator,
                      optimizer_S,
                      optimizer_D,
                      loss,
                      device,
                      save_dir,
                      clip_value=0.01,
                      n_critic=5,
                      )

    trainer.fit(n_epochs,
                trainloader_T, trainloader_S,
                valloader_T, valloader_S,)
    
    wandb.finish()

if __name__ == "__main__":
    main(train_dataroot="Compression-GAN/image/data/train",
         val_dataroot="Compression-GAN/image/data/val",
         imagesize=64,
         pretrained_model="resnet18",
         lr=2e-3,
         n_epochs=100,
         batch_size=8,
         device="cuda:0",
         save_dir="Compression-GAN/image")
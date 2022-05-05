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
        pretrained_model,
        lr,
        n_epochs,
        batch_size,
        device,
        save_dir,
        ckpt_path=None,):
        
    # Configure data loader
    os.makedirs(train_dataroot, exist_ok=True)
    os.makedirs(val_dataroot, exist_ok=True)
    trainloader_T = torch.utils.data.DataLoader(MNIST(dataroot=train_dataroot,
                                                    transform=transforms.Compose(
                                                    [transforms.ToTensor(), 
                                                    transforms.Normalize([0.5], [0.5])]
                                            )),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            )
    trainloader_S = torch.utils.data.DataLoader(MNIST(dataroot=train_dataroot,
                                                    transform=transforms.Compose(
                                                    [transforms.ToTensor(), 
                                                    transforms.Normalize([0.5], [0.5])]
                                            )),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            )
    valloader_T = torch.utils.data.DataLoader(MNIST(dataroot=val_dataroot,
                                                    transform=transforms.Compose(
                                                    [transforms.ToTensor(), 
                                                    transforms.Normalize([0.5], [0.5])]
                                            ), train=False),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            )
    valloader_S = torch.utils.data.DataLoader(MNIST(dataroot=val_dataroot,
                                                    transform=transforms.Compose(
                                                    [transforms.ToTensor(), 
                                                    transforms.Normalize([0.5], [0.5])]
                                            ), train=False),
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
    trainer = Trainer(
                      teacher, 
                      student,
                      discriminator,
                      optimizer_S,
                      optimizer_D,
                      loss,
                      save_dir,
                      ckpt_path,
                      )

    trainer.fit(n_epochs,
                trainloader_T, trainloader_S,
                valloader_T, valloader_S,)

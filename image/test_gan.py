import argparse
import os
import numpy as np
import math
import sys
import wandb
from itertools import cycle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models import classifier

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension") #28
parser.add_argument("--channels", type=int, default=3, help="number of image channels") #1
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Teacher(nn.Module):
    """
    A Teacher model giving the latent represntation of 
    input using pretrained model.
    """
    output_dim=32
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

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(opt.latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.shape[0], *img_shape)
#         return img

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

    
# ----------
#  Training
# ----------

def train():
    batches_done = 0
    for i, (data_T, data_S) in enumerate(zip(cycle(trainloader_T), trainloader_S)):
        imgs_T, imgs_S = data_T[0].cuda(), data_S[0].cuda()
        # Configure input
        with torch.no_grad():
            real_feat = teacher(imgs_T)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_feat = student(imgs_S).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_feat)) + torch.mean(discriminator(fake_feat))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_feat = student(imgs_S)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_feat))

            loss_G.backward()
            optimizer_G.step()

            wandb.log({
                "student-loss": loss_G.item(),
                "critic-loss": loss_D.item(),
            })

        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, batches_done % len(trainloader_T), len(trainloader_T), loss_D.item(), loss_G.item())
    )

def val():
    teacher.eval()
    student.eval()
    # ---------------------
    #  Downstream task
    # ---------------------
    classifier_T = classifier(1000, 10).cuda()
    optimizer_T = torch.optim.Adam(classifier_T.parameters(), lr=0.001)
    classifier_S = classifier(1000, 10).cuda()
    optimizer_S = torch.optim.Adam(classifier_S.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss().cuda()

    # ---Training---
    for epoch in range(10):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader_T):
            img, labels = data[0].cuda(), data[1].cuda()

            with torch.no_grad():
                feature_T = teacher(img)

            with torch.no_grad():
                feature_S = student(img)

            # ---TEACHER---
            optimizer_T.zero_grad()

            out_T = classifier_T(feature_T)
            loss_T = criterion(out_T, labels)
            loss_T.backward()
            optimizer_T.step()

            # ---STUDENT---
            optimizer_S.zero_grad()

            out_S = classifier_S(feature_S)
            loss_S = criterion(out_S, labels)
            loss_S.backward()
            optimizer_S.step()

    print('Finished Training Evaluator')

    correct_T, correct_S, total = 0, 0, 0
    for i, data in enumerate(valloader):

        img, labels = data[0].cuda(), data[1].cuda()

        with torch.no_grad():
            feature_T = teacher(img)

        with torch.no_grad():
            feature_S = student(img)

        # ---Teacher---
        out_T = classifier_T(feature_T)
        _, predicted_T = torch.max(out_T, 1)
        correct_T += (predicted_T == labels).sum().item()

        # ---Student---
        out_S = classifier_S(feature_S)
        _, predicted_S = torch.max(out_S, 1)
        correct_S += (predicted_S == labels).sum().item()

        total += labels.size(0)

    wandb.log({"Teacher Accuracy": correct_T/total,
               "Student Accuracy": correct_S/total})


    print(
        f"[VAL] - Teacher Accuracy : {correct_T/total}, Student Accuracy : {correct_S/total}"
    )

if __name__ == "__main__":
    # Initialize generator and discriminator
    teacher = Teacher()
    student = Student()
    discriminator = Discriminator(1000)

    if cuda:
        teacher.cuda()
        student.cuda()
        discriminator.cuda()

    # Configure data loader
    os.makedirs("../../data/cifar-10", exist_ok=True)
    trainloader_T = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(opt.img_size),
                                transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])),
                                batch_size=opt.batch_size,
                                shuffle=False,
                                        )
    trainloader_S = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(opt.img_size),
                                transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])),
                                batch_size=opt.batch_size,
                                shuffle=True,
                                        )
    valloader = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(opt.img_size),
                                transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])),
                                batch_size=opt.batch_size,
                                shuffle=True,
                                        )


    # Optimizers
    optimizer_G = torch.optim.RMSprop(student.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    wandb.init(project="Compression-GAN", entity="harsh1729")

    for epoch in range(opt.n_epochs):
        train()
        if (epoch+1) % 5 == 0:
            val()
        
    wandb.finish()
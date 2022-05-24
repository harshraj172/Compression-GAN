from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import wandb
import os
from itertools import cycle
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class Teacher(nn.Module):
    output_dim=1000
    def __init__(self, pretrained_model="resnet101"):
        super(Teacher, self).__init__()
        self.model = self.load_pretrained(pretrained_model)

    def load_pretrained(self, name):
        resnet101 = models.resnet101(pretrained=True, progress=False)
        for param in resnet101.parameters():
            param.requires_grad = False
        return resnet101

    def forward(self, x):
        x = self.model(x)
        return x

class Student(nn.Module):
    output_dim=1000
    def __init__(self, pretrained_model="resnet18"):
        super(Student, self).__init__()
        self.model = self.load_pretrained(pretrained_model)
        
    def load_pretrained(self, name):
        resnet18 = models.resnet18(pretrained=False, progress=False)
        return resnet18
    
    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

                        nn.Linear(1000, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2, inplace=True),
            
                        nn.Linear(512, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2, inplace=True),
            
                        nn.Linear(512, 1),
                        )

    def forward(self, x):
        validity = self.model(x)
        return validity
        

class KD_GAN():
    def __init__(self, device, 
                       batch_size,
                       latent_size, 
                       discriminator, 
                       teacher,
                       student,
                       trainloader_T,
                       trainloader_S,
                       valloader,
                       ckpt_path=None,):
        super(KD_GAN, self).__init__()
        self.discriminator = discriminator
        self.teacher = teacher 
        self.student = student
        self.trainloader_T = trainloader_T
        self.trainloader_S = trainloader_S
        self.valloader = valloader
        self.device = device
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.opt_g = torch.optim.Adam(self.student.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        if ckpt_path is not None:
            self.student, self.discriminator = self.load_checkpoint(self.student, self.discriminator, 
                                                                    ckpt_path)
            print(f"Loaded model from checkpoint: {ckpt_path}!!!")
            
    def train_discriminator(self, real_embs, images_S):
        # Clear discriminator gradients
        self.opt_d.zero_grad()
     
        # Pass real images through discriminator
        real_preds = self.discriminator(real_embs)
        real_targets = torch.ones(real_embs.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy_with_logits(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        fake_embs = self.student(images_S)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_embs.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_embs)
        fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        self.opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, images_S):
        # Clear generator gradients
        self.opt_g.zero_grad()
        
        # Generate fake images
        fake_embs = self.student(images_S)
        
        # Try to fool the discriminator
        preds = self.discriminator(fake_embs)
        targets = torch.ones(preds.size(0), 1, device=self.device)
        loss = F.binary_cross_entropy_with_logits(preds, targets)
        
        # Update generator weights
        loss.backward()
        self.opt_g.step()
        
        return loss.item()
    
    def evaluate(self, feat_model, trainloader, valloader):
        classifier = nn.Sequential(
                        # in: 3 x 64 x 64

                        nn.Linear(1000, 10),
                        ).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(self.device)

        # ---Training---
        for epoch in range(15):  # loop over the dataset multiple times

            for i, (imgs, labels) in enumerate(trainloader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    features = feat_model(imgs)
                    
                # ---TEACHER---
                optimizer.zero_grad()

                out = classifier(features)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

        print('Finished Training Evaluator')
        
        # ---Validating---
        correct, total = 0, 0
        for i, (imgs, labels) in enumerate(valloader):
            
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = feat_model(imgs)
                
            # ---Teacher---
            with torch.no_grad():
                out = classifier(features)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)
        
        accuracy = correct/total
        return accuracy
            
        
    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
    
    def load_checkpoint(self, student, discriminator,
                        ckpt_path):
        checkpoint = torch.load(ckpt_path)
        student.load_state_dict(checkpoint['student_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_dict'])
        return student, discriminator
    
    def fit(self, epochs, lr, start_idx=1):
        torch.cuda.empty_cache()
        
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
                
        for epoch in range(epochs):
            for i, ((images_T, _), (images_S, _)) in tqdm(enumerate(zip(cycle(self.trainloader_T), self.trainloader_S))):
                with torch.no_grad():
                    real_embs = self.teacher(images_T.to(self.device))

                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_embs, images_S.to(self.device))
                # Train generator
                loss_g = self.train_generator(images_S.to(self.device))
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
            wandb.log({'student-loss': loss_g,
                       'discriminator-loss': loss_d})
            
            if epoch % 10 == 0:
                accuracy_T = self.evaluate(self.teacher, self.trainloader_T, self.valloader)
                accuracy_S = self.evaluate(self.student, self.trainloader_T, self.valloader)
                print(f"[VAL] Teacher Accuarcy: {accuracy_T}, Student Accuracy: {accuracy_S}")
                wandb.log({'teacher-accuracy': accuracy_T,
                           'student-accuracy': accuracy_S})
                
            self.save_checkpoint({
            'student_dict': self.student.state_dict(),
            'discriminator_dict': self.discriminator.state_dict(),
            }, filename=f"Compression-GAN/W-Gan/kdgan2_acc-t_{accuracy_T}_acc-s_{accuracy_S}.pt")
            
        return losses_g, losses_d, real_scores, fake_scores


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2048
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    # Configure data loader
    os.makedirs("../../data/cifar-10", exist_ok=True)
    trainloader_T = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats)])),
                                batch_size=batch_size,
                                shuffle=True,
                                        )
    trainloader_S = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=True,
                                download=False,
                                transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats)])),
                                batch_size=batch_size,
                                shuffle=False,
                                        )
    valloader = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats)])),
                                batch_size=batch_size,
                                shuffle=True,
                                        )
    
    # Initialize generator and discriminator
    teacher = Teacher().to(device)
    student = Student().to(device)
    discriminator = Discriminator().to(device)
    kd_gan = KD_GAN(teacher=teacher,
              student=student,
              discriminator=discriminator,
              trainloader_T=trainloader_T,
              trainloader_S=trainloader_S,
              valloader=valloader, 
              device=device,
              latent_size=128,
              batch_size=batch_size,
              ckpt_path=None)
    
    wandb.init(project="Compression-GAN", entity="harsh1729")
    # Train
    losses_g, losses_d, real_scores, fake_scores = kd_gan.fit(100, 0.0002)
        
    wandb.finish()
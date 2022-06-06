from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import wandb
import os
import random
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
    def __init__(self, pretrained_model="resnet18"):
        super(Teacher, self).__init__()
        self.model = self.load_pretrained(pretrained_model)

    def load_pretrained(self, name):
        if name=="resnet18":
            resnet18 = models.resnet18(pretrained=True, progress=False)
            model = nn.Sequential(*list(resnet18.children())[:-2])
        elif name=="vgg16":
            vgg16 = models.vgg16(pretrained=True, progress=False)
            layers = list(vgg16.features[:-1])
            layers += [nn.MaxPool2d(kernel_size=2, padding=1)]
            model = nn.Sequential(*layers)
            
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, x):
        x = self.model(x)
        return x

class Student(nn.Module):
    def __init__(self, pretrained_model="resnet18"):
        super(Student, self).__init__()
        self.model = self.load_pretrained(pretrained_model)

    def load_pretrained(self, name):
        if name=="resnet18":
            resnet18 = models.resnet18(pretrained=False, progress=False)
            model = nn.Sequential(*list(resnet18.children())[:-2])
        elif name=="vgg16":
            vgg16 = models.vgg16(pretrained=True, progress=False)
            layers = list(vgg16.features[:-1])
            layers += [nn.MaxPool2d(kernel_size=2, padding=1)]
            model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        x = self.model(x)
        return x
            
# class Student(nn.Module):
#     def __init__(self):
#         super(Student, self).__init__()
#         self.model = nn.Sequential(

#                         nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.MaxPool2d(2),

#                         nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.MaxPool2d(2),

#                         nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False),
#                         nn.ReLU(inplace=True),
#                         nn.MaxPool2d(2),
            
#                         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0, bias=False),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(inplace=True),
#                         )

#     def forward(self, x):
#         x = self.model(x)
#         return x
    
# class Student(nn.Module):
#     def __init__(self):
#         super(Student, self).__init__()
#         self.model = nn.Sequential(

#                         nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=0, bias=False),
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(inplace=True),
#                         nn.MaxPool2d(2),

#                         nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1, bias=False),
#                         nn.BatchNorm2d(256),
#                         nn.ReLU(inplace=True),
#                         nn.MaxPool2d(2),

#                         nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=0, bias=False),
#                         nn.BatchNorm2d(512),
#                         nn.ReLU(inplace=True),
#                         )

#     def forward(self, x):
#         x = self.model(x)
#         return x
    
# class Teacher(nn.Module):
#     def __init__(self, pretrained_model="resnet34"):
#         super(Teacher, self).__init__()
#         self.model = self.load_pretrained(pretrained_model)

#     def load_pretrained(self, name):
#         resnet34 = models.resnet34(pretrained=True, progress=False)
#         for param in resnet34.parameters():
#             param.requires_grad = False
#         return resnet34

#     def getActivation(self, name, activation):
#         # the hook signature
#         def hook(model, input, output):
#             activation[name] = output.detach()
#         return hook

#     def forward(self, x):
#         activation = {}
#         h = self.model.layer4[2].bn2.register_forward_hook(self.getActivation('bn2', activation))
#         out = self.model(x)
#         h.remove()
#         x = activation['bn2']
#         return x

# class Student(nn.Module):
#     def __init__(self, pretrained_model="resnet18"):
#         super(Student, self).__init__()
#         self.model = self.load_pretrained(pretrained_model)
        
#     def load_pretrained(self, name):
#         resnet18 = models.resnet18(pretrained=False, progress=False)
#         model = nn.Sequential(*list(resnet18.children())[:-2])
#         return model
    
#     def forward(self, x):
#         x = self.model(x)
#         return x

# class Student(nn.Module):
#     def __init__(self, pretrained_model="resnet18"):
#         super(Student, self).__init__()
#         self.model = self.load_pretrained(pretrained_model)
        
#     def load_pretrained(self, name):
#         resnet50 = models.resnet50(pretrained=False, progress=False)
#         return resnet50
    
#     def getActivation(self, name, activation):
#         # the hook signature
#         def hook(model, input, output):
#             activation[name] = output.detach()
#         return hook
    
#     def forward(self, x):
#         activation = {}
#         h = self.model.layer4[2].bn2.register_forward_hook(self.getActivation('bn2', activation))
#         out = self.model(x)
#         h.remove()
#         x = F.relu(activation['bn2'])
#         return x
    
# class Student(nn.Module):
#     def __init__(self, pretrained_model="resnet18"):
#         super(Student, self).__init__()
#         self.model = self.load_pretrained(pretrained_model)
        
#     def load_pretrained(self, name):
#         resnet18 = models.resnet18(pretrained=False, progress=False)
#         model = nn.Sequential(*list(resnet18.children())[:-2])
#         return model
    
#     def forward(self, x):
#         x = self.model(x)
#         x = F.relu(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

                        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(1024),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(2048),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(2048, 512, kernel_size=4, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        # out: 512 x 2 x 2

                        nn.Flatten(),
                        nn.Linear(2048, 1),
                        )

    def forward(self, x):
        x = self.model(x)
        return x
        

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
                       lr,
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
        
        # Create optimizers
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_g = torch.optim.Adam(self.student.parameters(), lr=lr, betas=(0.5, 0.999))
        
        if ckpt_path is not None:
            self.student, self.discriminator = self.load_checkpoint(self.student, self.discriminator, 
                                                                    ckpt_path)
            print(f"Loaded model from checkpoint: {ckpt_path}!!!")
            
    def train(self, image_T, images_S):
        real_label, fake_label = 1, 0
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # Clear discriminator gradients
        self.opt_d.zero_grad()
        
        # Get teacher's prediction(real embeddings)
        with torch.no_grad():
            real_embs = self.teacher(image_T).detach()
        
        # format batch
        b_size = real_embs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
        
        # Forward pass real batch through D
        output = self.discriminator(real_embs).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = F.binary_cross_entropy_with_logits(output, label)
        
        # Calculate gradients for D in backward pass
        errD_real.backward()
        real_preds = output.mean().item()
        
        #-------------------------------------------------------------
        
        # Get student's prediction (fake embeddings)
        fake_embs = self.student(images_S)
        
        # Generate fake image batch with G
        label.fill_(fake_label)
        
        # Classify all fake batch with D
        output = self.discriminator(fake_embs.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = F.binary_cross_entropy_with_logits(output, label)
        
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        
        fake_preds1 = output.mean().item()
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.opt_d.step()
        

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if random.random() > 0.5:
            self.opt_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.discriminator(fake_embs).view(-1)

            # Calculate G's loss based on this output
            errG = F.binary_cross_entropy_with_logits(output, label)

            # Calculate gradients for G
            errG.backward()
            fake_preds2 = output.mean().item()

            # Update G
            self.opt_g.step()
        else:
            return errD.item(), real_preds, errD_real.item(), fake_preds1,\
        errD_fake.item(), None, None
        
        return errD.item(), real_preds, errD_real.item(), fake_preds1,\
    errD_fake.item(), errG.item(), fake_preds2

    
    def evaluate(self, feat_model, trainloader, valloader):
        classifier = nn.Sequential(
                        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(inplace=True),

                        nn.Flatten(),

                        nn.Linear(1024, 10),
                        ).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(self.device)

        # ---Training---
        for epoch in range(15):  # loop over the dataset multiple times

            for i, (imgs, labels) in enumerate(trainloader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    features = feat_model(imgs).detach()
                    
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
                features = feat_model(imgs).detach()
                
            # ---Teacher---
            with torch.no_grad():
                out = classifier(features).detach()
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
    
    def fit(self, epochs, start_idx=1):
        torch.cuda.empty_cache()
        
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        for epoch in range(epochs):
            for i, ((images_T, _), (images_S, _)) in tqdm(enumerate(zip(cycle(self.trainloader_T), self.trainloader_S))):
                
                # Train 
                errD, real_preds, errD_real, fake_preds1, errD_fake,\
                errG, fake_preds2 = self.train(images_T.to(self.device), images_S.to(self.device))
                
            
            # Log losses & scores (last batch)
            print(f"Epoch [{epoch+1}/{epochs}] , loss-g: {errG} loss-d: {errD} || real_preds: {real_preds} errD_real: {errD_real} || fake_preds1: {fake_preds1} errD_fake: {errD_fake} || fake_preds2: {fake_preds2}")
            wandb.log({'student-loss': errG,
                       'discriminator-loss': errD})
            
            if epoch % 10 == 0:
                accuracy_T = self.evaluate(self.teacher, self.trainloader_T, self.valloader)
                accuracy_S = self.evaluate(self.student, self.trainloader_T, self.valloader)
                print(f"[VAL] Teacher Accuracy: {accuracy_T}, Student Accuracy: {accuracy_S}")
                wandb.log({'teacher-accuracy': accuracy_T,
                           'student-accuracy': accuracy_S})
                
            # self.save_checkpoint({
            # 'student_dict': self.student.state_dict(),
            # 'discriminator_dict': self.discriminator.state_dict(),
            # }, filename=f"Compression-GAN/W-Gan/kdgan1_acc-t_{accuracy_T}_acc-s_{accuracy_S}.pt")
            


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    # Configure data loader
    os.makedirs("../../data/cifar-10", exist_ok=True)
    trainloader_T = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(84),
                                transforms.RandomCrop(84, padding=4, padding_mode="reflect"),
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
                                transforms.Resize(84),
                                transforms.RandomCrop(84, padding=4, padding_mode="reflect"),
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
                                transforms.Resize(84),
                                transforms.RandomCrop(84, padding=4, padding_mode="reflect"),
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
              lr=1e-5,
              ckpt_path=None)
    
    wandb.init(project="Compression-GAN", entity="harsh1729")
    # Train
    kd_gan.fit(epochs=100)
        
    wandb.finish()

import os 
import wandb
from itertools import cycle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from models import classifier
from utils import reduce_dim

class Trainer():
    def __init__(self,
                      num_classes,
                      teacher, 
                      output_dim,
                      student,
                      discriminator,
                      optimizer_S,
                      optimizer_D,
                      criterion,
                      device,
                      save_dir,
                      clip_value,
                      n_critic,
                      ):
        super(Trainer, self).__init__()
        self.num_classes = num_classes
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.teacher = teacher
        self.output_dim = output_dim    
        self.student = student
        self.discriminator = discriminator
        self.optimizer_S = optimizer_S
        self.optimizer_D = optimizer_D
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir

    def train(self, dataloader_T, dataloader_S):
        self.student.train()
        self.discriminator.train()

        Tensor = torch.cuda.FloatTensor
        for i, (data_T, data_S) in enumerate(zip(cycle(dataloader_T), dataloader_S)):
            img_T, img_S = data_T[0].to(self.device), data_S[0].to(self.device)

            with torch.no_grad():
                real_emb = self.teacher(img_T)
            real_emb = reduce_dim(real_emb)

            # Generate a batch of images
            gen_emb = self.student(img_S).detach()
            
            # ---------------------
            #  Loss Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()

            d_loss = -torch.mean(self.discriminator(real_emb)) + torch.mean(self.discriminator(gen_emb))

            # ---------------------
            #  Backpropagate Discriminator
            # ---------------------
            d_loss.backward()
            self.optimizer_D.step()

            # clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
                
            # ---------------------
            #  Loss Generator
            # ---------------------
            self.optimizer_S.zero_grad()
            # generate a batch of fake images
            gen_emb = self.student(img_S)
            # Adversarial loss
            g_loss = -torch.mean(self.discriminator(gen_emb))
                
            # ---------------------
            #  Backpropagate Generator
            # ---------------------    
            g_loss.backward()
            self.optimizer_S.step()
            
            if (i+1) % 100 == 0:
                wandb.log({"train--d_loss": d_loss.item(),
                           "train--g_loss": g_loss.item()})

        print(
            f"[TRAIN] Batch: {i}/{len(dataloader_T)}, D loss: {d_loss.item()}, G loss: {g_loss.item()}"
        )

    def val(self, trainloader, valloader):
        self.student.eval()
        self.discriminator.eval()
        
        # ---------------------
        #  Downstream task
        # ---------------------
        classifier_T = classifier(self.output_dim, self.num_classes).to(self.device)
        optimizer_T = Adam(classifier_T.parameters(), lr=0.001)
        classifier_S = classifier(self.output_dim, self.num_classes).to(self.device)
        optimizer_S = Adam(classifier_S.parameters(), lr=0.001)
        
        criterion = nn.CrossEntropyLoss().to(self.device)

        # ---Training---
        for epoch in range(10):  # loop over the dataset multiple times

            for i, data in enumerate(trainloader):
                img, labels = data[0].to(self.device), data[1].to(self.device)

                with torch.no_grad():
                    feature_T = self.teacher(img)
                feature_T = reduce_dim(feature_T)
                
                with torch.no_grad():
                    feature_S = self.student(img)

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
            
            img, labels = data[0].to(self.device), data[1].to(self.device)

            with torch.no_grad():
                feature_T = self.teacher(img)
            feature_T = reduce_dim(feature_T)

            with torch.no_grad():
                feature_S = self.student(img)
                
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

    def load_checkpoint(self, student, discriminator,
                        optimizer_S, optimizer_D, 
                        ckpt_path):
        checkpoint = torch.load(ckpt_path)
        student.load_state_dict(checkpoint['student_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_dict'])
        optimizer_S.load_state_dict(checkpoint['optimizer_S_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_dict'])
        return student, discriminator, optimizer_S, optimizer_D

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def fit(self, n_epochs,
            trainloader_T, trainloader_S,
            valloader_T, valloader_S,):
        
        if os.path.isfile(f"{self.save_dir}/model.pt"):
            ckpt_path = f"{self.save_dir}/model.pt" 
            self.student, self.discriminator, self.optimizer_S, self.optimizer_D =\
                self.load_checkpoint(self.student, self.discriminator,
                                self.optimizer_S, self.optimizer_D, 
                                ckpt_path)
            print(f"checkpoint - {ckpt_path} - Loaded")
                                
        for epoch in range(n_epochs):
            print(f"EPOCH: {epoch}/{n_epochs}")
            self.train(trainloader_T, trainloader_S)
            self.val(trainloader_T, valloader_S)
            print()
            
            self.save_checkpoint({
            'student_dict': self.student.state_dict(),
            'discriminator_dict': self.discriminator.state_dict(),
            'optimizer_S_dict': self.optimizer_S.state_dict(),
            'optimizer_D_dict': self.optimizer_D.state_dict(),
            }, filename=f"{self.save_dir}/model.pt")

from itertools import cycle

import torch
from torch.autograd import Variable

class Trainer():
    def __init__(self,teacher, 
                      student,
                      discriminator,
                      optimizer_S,
                      optimizer_D,
                      loss,
                      device,
                      save_dir,
                      ckpt_path,):
        super(Trainer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.discriminator = discriminator
        self.optimizer_S = optimizer_S
        self.optimizer_D = optimizer_D
        self.loss = loss
        self.device = device
        self.save_dir = save_dir
        self.ckpt_path = ckpt_path

    def train(self, dataloader_T, dataloader_S):
        self.student.train()
        self.discriminator.train()

        Tensor = torch.cuda.FloatTensor
        for i, (data_T, data_S) in enumerate(zip(cycle(dataloader_T), dataloader_S)):
            img_T, img_S = data_T[0].to(self.device), data_S[0].to(self.device)

            # Adversarial ground truths
            valid = Variable(Tensor(img_T.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img_T.size(0), 1).fill_(0.0), requires_grad=False)

            with torch.no_grad():
                real_emb = self.teacher(img_T)

            # -----------------
            #  Train Student
            # -----------------
            self.optimizer_S.zero_grad()

            # Generate a batch of images
            gen_emb = self.student(img_S)

            # Loss measures Student's ability to fool the discriminator
            g_loss = self.loss(self.discriminator(gen_emb), valid)

            g_loss.backward()
            self.optimizer_S.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.loss(self.discriminator(real_emb), valid)
            fake_loss = self.loss(self.discriminator(gen_emb.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

        print(
            f"[TRAIN] Batch: {i}/{len(dataloader_T)}, D loss: {d_loss.item()}, G loss: {g_loss.item()}"
        )

    def val(self, dataloader_T, dataloader_S):
        self.student.eval()
        self.discriminator.eval()

        Tensor = torch.cuda.FloatTensor
        for i, (data_T, data_S) in enumerate(zip(cycle(dataloader_T), dataloader_S)):
            img_T, img_S = data_T[0].to(self.device), data_S[0].to(self.device)

            # Adversarial ground truths
            valid = Variable(Tensor(img_T.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img_T.size(0), 1).fill_(0.0), requires_grad=False)

            with torch.no_grad():
                real_emb = self.teacher(img_T)

            # -----------------
            #  Train Student
            # -----------------
            # Generate a batch of data
            gen_emb = self.student(img_S)

            # Loss measures Student's ability to fool the discriminator
            g_loss = self.loss(self.discriminator(gen_emb), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.loss(self.discriminator(real_emb), valid)
            fake_loss = self.loss(self.discriminator(gen_emb.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
        
        print(
            f"[VAL] Batch: {i}/{len(dataloader_T)}, D loss: {d_loss.item()}, G loss: {g_loss.item()}"
        )
        print(f"Real Loss: {real_loss}")
        print(f"Fake Loss: {fake_loss}")

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
        
        if self.ckpt_path:
            self.student, self.discriminator, self.optimizer_S, self.optimizer_D =\
                load_checkpoint(self.student, self.discriminator,
                                self.optimizer_S, self.optimizer_D, 
                                self.ckpt_path)
                                
        for epoch in range(n_epochs):
            print(f"EPOCH: {epoch}/{n_epochs}")
            self.train(trainloader_T, trainloader_S)
            self.val(valloader_T, valloader_S)
            print()
            
            self.save_checkpoint({
            'student_dict': self.student.state_dict(),
            'discriminator_dict': self.discriminator.state_dict(),
            'optimizer_S_dict': self.optimizer_S.state_dict(),
            'optimizer_D_dict': self.optimizer_D.state_dict(),
            }, filename=f"{self.save_dir}/model.pt")

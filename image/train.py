import torch


class Trainer():
    def __init__(self,teacher, 
                      student,
                      discriminator,
                      optimizer_S,
                      optimizer_D,
                      loss,
                      save_dir,
                      ckpt_path,):
        super(Trainer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.discriminator = discriminator
        self.optimizer_S = optimizer_S
        self.optimizer_D = optimizer_D
        self.loss = loss
        self.save_dir = save_dir
        self.ckpt_path = ckpt_path

    def train(dataloader_T, dataloader_S):
        self.student.train()
        self.discriminator.train()

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for i, (data_T, data_S) in enumerate(zip(cycle(dataloader_T), dataloader_S)):
            data_T, data_S = data_T.to(device), data_S.to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(data_T.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data_T.size(0), 1).fill_(0.0), requires_grad=False)

            with torch.no_grad:
                real_emb = self.teacher(data_T)

            # -----------------
            #  Train Student
            # -----------------
            optimizer_S.zero_grad()

            # Generate a batch of images
            gen_emb = self.student(data_S)

            # Loss measures Student's ability to fool the discriminator
            g_loss = loss(discriminator(gen_emb), valid)

            g_loss.backward()
            optimizer_S.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(self.discriminator(real_emb), valid)
            fake_loss = loss(self.discriminator(gen_emb.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[TRAIN][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

    def val(dataloader_T, dataloader_S):
        self.student.eval()
        self.discriminator.eval()

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for i, (data_T, data_S) in enumerate(zip(cycle(self.dataloader_T), self.dataloader_S)):
            data_T, data_S = data_T.to(device), data_S.to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(data_T.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data_T.size(0), 1).fill_(0.0), requires_grad=False)

            with torch.no_grad:
                real_emb = self.teacher(data_T)

            # -----------------
            #  Train Student
            # -----------------
            # Generate a batch of data
            gen_emb = self.student(data_S)

            # Loss measures Student's ability to fool the discriminator
            g_loss = loss(self.discriminator(gen_emb), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(self.discriminator(real_emb), valid)
            fake_loss = loss(self.discriminator(gen_emb.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            print(
                "[VAL][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(valloader), d_loss.item(), g_loss.item())
            )

    def load_checkpoint(student, discriminator,
                        optimizer_S, optimizer_D, 
                        ckpt_path):
        checkpoint = torch.load(ckpt_path)
        student.load_state_dict(checkpoint['student_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_dict'])
        optimizer_S.load_state_dict(checkpoint['optimizer_S_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_dict'])
        return student, discriminator, optimizer_S, optimizer_D

    def save_checkpoint(state, filename):
        torch.save(state, filename)

    def fit(n_epochs,
            trainloader_T, trainloader_S,
            valloader_T, valloader_S,):
        
        if self.ckpt_path:
            self.student, self.discriminator, self.optimizer_S, self.optimizer_D =\
                load_checkpoint(self.student, self.discriminator,
                                self.optimizer_S, self.optimizer_D, 
                                self.ckpt_path)
                                
        for epoch in range(n_epochs):
            self.train(trainloader_T, trainloader_S)
            self.val(valloader_T, valloader_S)

            self.save_checkpoint({
            'student_dict': self.student.state_dict(),
            'discriminator_dict': self.discriminator.state_dict(),
            'optimizer_S_dict': self.optimizer_S.state_dict(),
            'optimizer_D_dict': self.optimizer_D.state_dict(),
            }, filename=save_dir)

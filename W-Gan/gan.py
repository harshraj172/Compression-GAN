from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import wandb
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_size=128):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
                        # in: latent_size x 1 x 1

                        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(True),
                        # out: 512 x 4 x 4

                        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(True),
                        # out: 256 x 8 x 8

                        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        # out: 128 x 16 x 16

                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True),
                        # out: 64 x 32 x 32

                        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.Tanh()
                        # out: 3 x 64 x 64
                    )

    def forward(self, x):
        x = self.model(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                        # in: 3 x 64 x 64

                        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        # out: 64 x 32 x 32

                        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        # out: 128 x 16 x 16

                        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        # out: 256 x 8 x 8

                        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        # out: 512 x 4 x 4

                        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
                        # out: 1 x 1 x 1

                        nn.Flatten(),
                        nn.Sigmoid())

    def forward(self, x):
        validity = self.model(x)

        return validity
        

class GAN():
    def __init__(self, device, 
                       batch_size,
                       latent_size, 
                       discriminator, 
                       generator,
                       trainloader):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator 
        self.trainloader = trainloader
        self.device = device
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
        self.sample_dir = 'generated'
        os.makedirs(self.sample_dir, exist_ok=True)

    def train_discriminator(self, real_images, opt_d):
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, opt_g):
        # Clear generator gradients
        opt_g.zero_grad()
        
        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        
        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(self.batch_size, 1, device=self.device)
        loss = F.binary_cross_entropy(preds, targets)
        
        # Update generator weights
        loss.backward()
        opt_g.step()
        
        return loss.item()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        
    def denorm(self, img_tensors):
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        return img_tensors * stats[1][0] + stats[0][0]

    def save_samples(self, index, latent_tensors, show=True):
        fake_images = self.generator(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(self.denorm(fake_images), os.path.join(self.sample_dir, fake_fname), nrow=8)
        print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

    def fit(self, epochs, lr, start_idx=1):
        torch.cuda.empty_cache()
        
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        for epoch in range(epochs):
            for i, (real_images, _) in tqdm(enumerate(self.trainloader)):
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images.to(self.device), opt_d)
                # Train generator
                loss_g = self.train_generator(opt_g)
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

            # Save generated images
            self.save_samples(epoch+start_idx, self.fixed_latent, show=False)
            self.save_checkpoint({
            'generator_dict': self.generator.state_dict(),
            'discriminator_dict': self.discriminator.state_dict(),
            'optimizer_g_dict': opt_g.state_dict(),
            'optimizer_d_dict': opt_d.state_dict(),
            }, filename=f"Compression-GAN/W-Gan/gan.pt")
            
        return losses_g, losses_d, real_scores, fake_scores


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    # Configure data loader
    os.makedirs("../../data/cifar-10", exist_ok=True)
    trainloader = torch.utils.data.DataLoader(
                        datasets.CIFAR10(
                                "../../data/cifar-10",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.Resize(64),
                                transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(*stats)])),
                                batch_size=batch_size,
                                shuffle=False,
                                        )
    
    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    gan = GAN(generator=generator, 
              discriminator=discriminator,
              trainloader=trainloader,
              device=device,
              latent_size=128,
              batch_size=batch_size,)
    
    wandb.init(project="Vanilla-GAN", entity="harsh1729")
    # Train
    losses_g, losses_d, real_scores, fake_scores = gan.fit(100, 0.0002)
        
    wandb.finish()
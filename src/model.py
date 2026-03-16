"""
model.py — Generator and Discriminator architectures for Cloud Removal GAN
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Encoder-Decoder CNN Generator.
    Maps a cloud-obscured satellite image (3×64×64) to a clean image (3×64×64).
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # → 64×32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # → 128×16×16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Decoder
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # → 64×32×32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # → 3×64×64
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    """
    PatchGAN-style Discriminator.
    Classifies whether a satellite image is real (clean) or fake (GAN-generated).
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # → 64×32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # → 128×16×16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


def weights_init(m):
    """Apply Xavier weight initialization to Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    dummy = torch.randn(4, 3, 64, 64).to(device)
    print("Generator output:", G(dummy).shape)
    print("Discriminator output:", D(dummy).shape)

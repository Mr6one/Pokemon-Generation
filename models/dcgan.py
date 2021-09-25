import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_vector_size, feature_map_size):
        super().__init__()

        self.lvs = latent_vector_size
        self.fms = feature_map_size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.lvs, self.fms * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.fms * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.fms * 8, self.fms * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fms * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.fms * 4, self.fms * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fms * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.fms * 2, self.fms, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fms),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.fms, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, feature_map_size):
        super().__init__()

        self.fms = feature_map_size

        self.main = nn.Sequential(
            nn.Conv2d(3, self.fms, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fms, self.fms * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fms * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fms * 2, self.fms * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fms * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fms * 4, self.fms * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fms * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fms * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
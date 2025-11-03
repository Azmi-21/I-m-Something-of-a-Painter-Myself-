"""
Minimal DCGAN Generator / Discriminator implementations.
"""
import torch
import torch.nn as nn

# cap channels at 512 to avoid excessive memory usage in deeper layers
def _cap(ch):
    return min(ch, 512)

class DCGenerator256(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        """
        Args:
            nz: size of the latent z vector (input to the generator)
            ngf: base number of generator feature maps
            nc: number of channels in the output image (e.g., 3 for RGB)
        """
        super().__init__()
        # Start: (nz) x 1 x 1 -> 4x4, then double until 256x256
        self.net = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose2d(nz, _cap(ngf * 16), 4, 1, 0, bias=False),
            nn.BatchNorm2d(_cap(ngf * 16)),
            nn.ReLU(True),
            # 4 -> 8
            nn.ConvTranspose2d(_cap(ngf * 16), _cap(ngf * 8), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ngf * 8)),
            nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose2d(_cap(ngf * 8), _cap(ngf * 4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ngf * 4)),
            nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose2d(_cap(ngf * 4), _cap(ngf * 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ngf * 2)),
            nn.ReLU(True),
            # 32 -> 64
            nn.ConvTranspose2d(_cap(ngf * 2), _cap(ngf), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ngf)),
            nn.ReLU(True),
            # 64 -> 128
            nn.ConvTranspose2d(_cap(ngf), _cap(ngf // 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ngf // 2)),
            nn.ReLU(True),
            # 128 -> 256
            nn.ConvTranspose2d(_cap(ngf // 2), nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class DCDiscriminator256(nn.Module):
    def __init__(self, nc: int = 3, ndf: int = 64):
        """
        Args:
            nc: number of channels in the input image (e.g., 3 for RGB)
            ndf: base number of discriminator feature maps
        """
        super().__init__()
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1
        self.net = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(ndf, _cap(ndf * 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ndf * 2)),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(_cap(ndf * 2), _cap(ndf * 4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ndf * 4)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(_cap(ndf * 4), _cap(ndf * 8), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ndf * 8)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(_cap(ndf * 8), _cap(ndf * 16), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ndf * 16)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(_cap(ndf * 16), _cap(ndf * 32), 4, 2, 1, bias=False),
            nn.BatchNorm2d(_cap(ndf * 32)),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 -> 1
            nn.Conv2d(_cap(ndf * 32), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1).squeeze(1)


def weights_init(m):
    """
    Custom weight initialization called on model.apply()
    Uses normal distribution with mean=0, std=0.02 for Conv layers
    and mean=1, std=0.02 for BatchNorm layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def build_dcgan256(nz: int = 100, nc: int = 3, ngf: int = 64, ndf: int = 64, device=None):
    """
    Builds and initializes DCGAN Generator and Discriminator for 256x256 images.
    Args:
        nz: size of the latent z vector (input to the generator)
        nc: number of channels in the images (e.g., 3 for RGB)
        ngf: base number of generator feature maps
        ndf: base number of discriminator feature maps
        device: torch device to place the models on (default: CPU)
    Returns:
        G: initialized DCGenerator256 model
        D: initialized DCDiscriminator256 model
    """
    device = device or torch.device("cpu")
    G = DCGenerator256(nz=nz, ngf=ngf, nc=nc).to(device)
    D = DCDiscriminator256(nc=nc, ndf=ndf).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D
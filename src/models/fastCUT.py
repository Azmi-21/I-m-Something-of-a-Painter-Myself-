import torch
import torch.nn as nn
import torch.nn.functional as F

# residual block used in the generator 
# this is a standard ResNet

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        self.use_norm = norm
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        if norm:
            self.bn1 = nn.InstanceNorm2d(out_ch)
            self.bn2 = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        if self.use_norm: out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_norm: out = self.bn2(out)
        return self.act(out + identity)


# generator network
# CUT/FastCUT uses: 1 downsample, N residual blocks, 1 upsample

class FastCUT_Generator(nn.Module):
    def __init__(self, in_ch=3, ngf=64):
        super().__init__()

        # encoder
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, ngf, 7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, padding=1, stride=2),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True)
        )

    
        blocks = []
        for _ in range(4):
            blocks.append(ResBlock(ngf*2, ngf*2))
        self.middle = nn.Sequential(*blocks)

        # decoder
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, in_ch, 7, padding=3),
            nn.Tanh()
        )

    def encode(self, x):
        f1 = self.down[:3](x)     # shallow layer
        f2 = self.down[3:](f1)    # deeper layer
        f3 = self.middle(f2)      # resblocks
        return {"f1": f1, "f2": f2, "f3": f3}

    def decode(self, z):
        return self.up(z)

    def forward(self, x, return_feats=False):
        feats = self.encode(x)
        out = self.decode(feats["f3"])
        if return_feats:
            return out, feats
        return out


# discriminator network (PatchGAN)

class FastCUT_Discriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), 
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, 1, 4, 1, 1) 
        )

    def forward(self, x):
        return self.net(x)


# mlp to project features for PatchNCE loss
class PatchMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


# patchNCE loss according to CUT paper
class PatchNCELoss(nn.Module):
    def __init__(self, temp=0.07, num_patches=256):
        super().__init__()
        self.temp = temp
        self.num = num_patches
        self.ce = nn.CrossEntropyLoss()

    def forward(self, q, k):
        """
        q, k: feature dict[layer_name] = [B,C,H,W]
        """
        total = 0
        n_layers = 0

        for name in q.keys():
            fq = q[name]
            fk = k[name]

            B, C, H, W = fq.shape
            HW = H * W

            # sample positions
            idx = torch.randint(0, HW, (B, self.num), device=fq.device)
            fq_flat = fq.view(B, C, HW)
            fk_flat = fk.view(B, C, HW)

            # gather patches
            q_p = torch.gather(fq_flat, 2, idx.unsqueeze(1).expand(-1, C, -1))  
            k_p = torch.gather(fk_flat, 2, idx.unsqueeze(1).expand(-1, C, -1))

            q_p = q_p.permute(0,2,1).reshape(B*self.num, C)
            k_p = k_p.permute(0,2,1).reshape(B*self.num, C)

            # all negatives from k
            negatives = fk_flat.permute(0,2,1).reshape(B*HW, C)

            logits = torch.mm(q_p, negatives.t()) / self.temp

            # positive indices
            batch_ids = torch.arange(B, device=fq.device).view(B,1).expand(-1,self.num)
            positive_indices = (batch_ids*HW + idx).reshape(-1)

            loss = self.ce(logits, positive_indices)
            total += loss
            n_layers += 1

        return total / n_layers


# create FastCUT model
def build_fastcut(ngf=64, ndf=64, device=None):
    device = device or torch.device("cpu")
    G = FastCUT_Generator(ngf=ngf).to(device)
    D = FastCUT_Discriminator(ndf=ndf).to(device)

    H = nn.ModuleDict({
        "f1": PatchMLP(ngf),
        "f2": PatchMLP(ngf*2),
        "f3": PatchMLP(ngf*2)
    }).to(device)

    return G, D, H

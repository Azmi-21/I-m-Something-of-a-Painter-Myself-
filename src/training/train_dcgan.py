import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from ..data.dataset import MonetPhotoDataset
from ..models.dcgan import build_dcgan256

def train(data_root, out_dir, epochs=50, batch_size=8, nz=100, lr=2e-4, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(out_dir, exist_ok=True)

    # dataset expects transforms that resize to 256 and normalize to [-1,1]
    dataset = MonetPhotoDataset(data_root, img_size=256)  # adjust API if needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    G, D = build_dcgan256(nz=nz, nc=3, ngf=64, ndf=64, device=device)
    criterion = nn.BCELoss()
    optimD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, nz, device=device)
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            imgs, _ = batch          # unpack (images, filenames)
            imgs = imgs.to(device)

            # update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # equivalent to minimizing BCE loss
            D.zero_grad()
            bsize = imgs.size(0)
            label = torch.full((bsize,), real_label, dtype=torch.float, device=device)
            output = D(imgs)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(bsize, nz, device=device)
            fake = G(noise.unsqueeze(-1).unsqueeze(-1))  # ensure shape (B, nz,1,1)
            label.fill_(fake_label)
            output = D(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimD.step()

            # update generator: maximize log(D(G(z)))
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake)
            errG = criterion(output, label)
            errG.backward()
            optimG.step()

            # print losses occasionally
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Loss_D: {(errD_real+errD_fake).item():.4f} Loss_G: {errG.item():.4f}")

        # save samples & checkpoint each epoch
        with torch.no_grad():
            sample = G(fixed_noise.unsqueeze(-1).unsqueeze(-1)).cpu()
            vutils.save_image((sample + 1) / 2.0, os.path.join(out_dir, f"epoch_{epoch+1:03d}.png"),
                              nrow=4, padding=2)
        torch.save({'G': G.state_dict(), 'D': D.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   os.path.join(out_dir, f"checkpoint_epoch_{epoch+1:03d}.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="path to dataset root")
    p.add_argument("--out_dir", default="outputs", help="where to save samples/checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--nz", type=int, default=100)
    args = p.parse_args()
    train(args.data_root, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, nz=args.nz)
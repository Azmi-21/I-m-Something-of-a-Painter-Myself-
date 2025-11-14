import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from ..data.dataset import MonetPhotoDataset
from ..models.fastCUT import build_fastcut, PatchNCELoss


def train(data_root, out_dir, epochs=100, batch_size=1, lr=0.0002, device=None, num_workers=4):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    dataset = MonetPhotoDataset(data_root, img_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # build FastCUT model to get Generator, Discriminator, and Projection Heads
    G, D, H = build_fastcut(ngf=64, ndf=64, device=device)
    nce_loss = PatchNCELoss(temp=0.07, num_patches=256)

    opt_G = torch.optim.Adam(list(G.parameters()) + list(H.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    gan_loss = nn.MSELoss()

    for epoch in range(epochs):
        for i, (photos, monets) in enumerate(loader):
            photos = photos.to(device)   # input X
            monets = monets.to(device)   # target Y

            # train discriminator
            D.zero_grad()
            fake_y = G(photos).detach()

            pred_real = D(monets)
            pred_fake = D(fake_y)

            loss_D = (gan_loss(pred_real, torch.ones_like(pred_real)) +
                      gan_loss(pred_fake, torch.zeros_like(pred_fake))) * 0.5

            loss_D.backward()
            opt_D.step()

            # train generator + H
            G.zero_grad()
            H.zero_grad()

            fake_y, feats_q = G(photos, return_feats=True)
            _, feats_k = G(photos, return_feats=True)  # input features for NCE

            # PatchNCE: q = fake, k = photo
            proj_q = {}
            proj_k = {}
            for k_name in feats_q.keys():
                if k_name in H:
                    B, C, Hh, Ww = feats_q[k_name].shape
                    proj_q[k_name] = H[k_name](feats_q[k_name]
                                               .permute(0,2,3,1)
                                               .reshape(-1, C)).reshape(B,Hh,Ww,-1).permute(0,3,1,2)
                    proj_k[k_name] = H[k_name](feats_k[k_name]
                                               .permute(0,2,3,1)
                                               .reshape(-1, C)).reshape(B,Hh,Ww,-1).permute(0,3,1,2)

            loss_NCE = nce_loss(proj_q, proj_k)
            loss_GAN = gan_loss(D(fake_y), torch.ones_like(pred_real))

            loss_G = loss_GAN + loss_NCE

            loss_G.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"[{epoch}/{epochs}] [{i}/{len(loader)}] "
                      f"D: {loss_D.item():.3f}  G: {loss_G.item():.3f}  NCE: {loss_NCE.item():.3f}")

        # save sample
        with torch.no_grad():
            out = G(photos[:4])
            vutils.save_image((out+1)/2, os.path.join(out_dir, f"epoch_{epoch+1:03}.png"))

        # save checkpoint
        ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "H": H.state_dict(),
            "optG": opt_G.state_dict(),
            "optD": opt_D.state_dict()
        }
        torch.save(ckpt, os.path.join(out_dir, f"fastcut_epoch_{epoch+1:03}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="outputs_fastcut")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    train(args.data_root, args.out_dir,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          num_workers=args.num_workers)

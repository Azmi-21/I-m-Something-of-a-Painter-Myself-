import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from ..data.dataset import PairedMonetPhotoDataset
from ..models.fastCUT import build_fastcut, PatchNCELoss


def random_crop(imgs, crop_size=128):
    """
    imgs: [B, C, H, W]  ->  random  crop_size x crop_size per image.
    If crop_size >= H/W, returns imgs unchanged.
    """
    B, C, H, W = imgs.shape
    if crop_size >= H or crop_size >= W:
        return imgs

    out = []
    for b in range(B):
        top = torch.randint(0, H - crop_size + 1, (1,)).item()
        left = torch.randint(0, W - crop_size + 1, (1,)).item()
        crop = imgs[b : b + 1, :, top : top + crop_size, left : left + crop_size]
        out.append(crop)
    return torch.cat(out, dim=0)


def train(
    data_root,
    out_dir,
    epochs=100,
    batch_size=1,
    lr=2e-4,
    device=None,
    num_workers=4,
    crop_size=128,
):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    os.makedirs(out_dir, exist_ok=True)

    # paired dataset: each item -> (monet, photo)
    dataset = PairedMonetPhotoDataset(data_root, img_size=256)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # build models
    G, D, H = build_fastcut(ngf=64, ndf=64, device=device)
    nce_loss_fn = PatchNCELoss(temp=0.07, num_patches=256)

    # --- OPTIMIZERS (TTUR) ---
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(list(G.parameters()) + list(H.parameters()), lr=lr * 0.5, betas=(0.5, 0.999))

    # LSGAN loss (as in CUT/FastCUT)
    gan_loss = nn.MSELoss()

    # use a fixed batch for visual tracking, this allows seeing progression on same inputs
    fixed_monet, fixed_photo = next(iter(loader))
    fixed_photo = fixed_photo.to(device)

    def step_lr_decay(optimizer, factor):
        for g in optimizer.param_groups:
            g["lr"] *= factor

    for epoch in range(epochs):

        if (epoch + 1) == 40:
            step_lr_decay(opt_D, 0.1)
            step_lr_decay(opt_G, 0.1)

        for i, (monets, photos) in enumerate(loader):
            photos = photos.to(device)   # domain X
            monets = monets.to(device)   # domain Y (target style)

            # crop randomly, this is done because CUT/FastCUT train on random crops
            # to encourage learning from local patches. Not doing this causes collapse
            photos_c = random_crop(photos, crop_size=crop_size)
            monets_c = random_crop(monets, crop_size=crop_size)

            # discriminator train step
            D.zero_grad()

            with torch.no_grad():
                fake_y = G(photos_c)  # X -> Y_hat

            pred_real = D(monets_c)
            pred_fake = D(fake_y)

            loss_D_real = gan_loss(pred_real, torch.ones_like(pred_real))
            loss_D_fake = gan_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            opt_D.step()

            # generator + projection heads train step
            G.zero_grad()
            H.zero_grad()

            # forward again so gradients flow through G
            fake_y = G(photos_c)  # [B,3,h,w]

            # NCE loss
            feats_k = G.encode(photos_c)   # dict[layer] -> [B,C,H,W]
            feats_q = G.encode(fake_y)     # dict[layer] -> [B,C,H,W]

            proj_q = {}
            proj_k = {}
            for lname, fmap_q in feats_q.items():
                if lname not in H:
                    continue
                fmap_k = feats_k[lname]

                B, C, Hh, Ww = fmap_q.shape

                # flatten spatial, project, reshape back to [B,C_out,H,W]
                q_flat = fmap_q.permute(0, 2, 3, 1).reshape(-1, C)
                k_flat = fmap_k.permute(0, 2, 3, 1).reshape(-1, C)

                z_q = H[lname](q_flat)
                z_k = H[lname](k_flat)

                C_out = z_q.shape[-1]
                proj_q[lname] = z_q.view(B, Hh, Ww, C_out).permute(0, 3, 1, 2)
                proj_k[lname] = z_k.view(B, Hh, Ww, C_out).permute(0, 3, 1, 2)

            loss_NCE = nce_loss_fn(proj_q, proj_k)

            # Identity loss: G should be close to identity on Monet inputs
            id_y = G(monets_c)
            lambda_id = 0.1  # will try between 0.05 and 0.5
            loss_id = torch.nn.functional.l1_loss(id_y, monets_c)

            # GAN loss for generator (fool D on fake Monet)
            pred_fake_for_G = D(fake_y)
            loss_GAN = gan_loss(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

            loss_G = loss_GAN + loss_NCE + lambda_id * loss_id

            loss_G.backward()
            opt_G.step()

            if i % 50 == 0:
                print(
                    f"[{epoch+1}/{epochs}] [{i}/{len(loader)}] "
                    f"D: {loss_D.item():.3f}  "
                    f"G: {loss_G.item():.3f}  "
                    f"NCE: {loss_NCE.item():.3f}"
                )
                print(f"... D_LR={opt_D.param_groups[0]['lr']:.2e}  G_LR={opt_G.param_groups[0]['lr']:.2e}")

        # Save sample images (same fixed input each epoch)
        with torch.no_grad():
            sample_fake = G(fixed_photo[:4].to(device))
            vutils.save_image(
                (sample_fake + 1) / 2,
                os.path.join(out_dir, f"epoch_{epoch+1:03}.png"),
                nrow=2,
            )

        # Save checkpoint
        ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "H": H.state_dict(),
            "optG": opt_G.state_dict(),
            "optD": opt_D.state_dict(),
            "epoch": epoch + 1,
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
    p.add_argument("--crop_size", type=int, default=128)
    args = p.parse_args()

    train(
        args.data_root,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
    )

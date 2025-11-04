import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from ..data.dataset import MonetPhotoDataset
from ..models.dcgan import build_dcgan256

def train(data_root, out_dir, epochs=50, batch_size=8, nz=100, lr=2e-4, device=None, n_critic=3):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    
    # Clear CUDA cache if using GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # dataset expects transforms that resize to 256 and normalize to [-1,1]
    dataset = MonetPhotoDataset(data_root, img_size=256)  # adjust API if needed
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))

    G, D = build_dcgan256(nz=nz, nc=3, ngf=64, ndf=64, device=device)
    criterion = nn.BCELoss()
    optimD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, nz, device=device)
    real_label = 0.9  # label smoothing for stability
    fake_label = 0.0

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            try:
                imgs, _ = batch          # unpack (images, filenames)
                imgs = imgs.to(device)
                bsize = imgs.size(0)

                # Train discriminator multiple times for stability
                for _ in range(n_critic):
                    D.zero_grad()
                    
                    # Real images
                    label = torch.full((bsize,), real_label, dtype=torch.float, device=device)
                    output_real = D(imgs)
                    errD_real = criterion(output_real, label)
                    errD_real.backward()
                    D_x = output_real.mean().item()  # monitor discriminator output on real

                    # Fake images
                    noise = torch.randn(bsize, nz, device=device)
                    fake = G(noise)  # generator already reshapes internally
                    label.fill_(fake_label)
                    output_fake = D(fake.detach())
                    errD_fake = criterion(output_fake, label)
                    errD_fake.backward()
                    D_G_z1 = output_fake.mean().item()  # monitor discriminator output on fake
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                    optimD.step()

                # Update generator once
                G.zero_grad()
                label.fill_(real_label)
                output = D(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()  # monitor after G update
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                optimG.step()

                # Print losses and discriminator outputs
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                          f"Loss_D: {(errD_real+errD_fake).item():.4f} Loss_G: {errG.item():.4f} "
                          f"D(x): {D_x:.3f} D(G(z)): {D_G_z1:.3f}/{D_G_z2:.3f}")
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"\nCUDA error at epoch {epoch+1}, batch {i}: {e}")
                    print("Clearing CUDA cache and continuing...")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Clear cache after each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # save samples each epoch
        try:
            with torch.no_grad():
                sample = G(fixed_noise).cpu()
                vutils.save_image((sample + 1) / 2.0, os.path.join(out_dir, f"epoch_{epoch+1:03d}.png"),
                                  nrow=4, padding=2)
        except Exception as e:
            print(f"Warning: Failed to save sample image: {e}")
        
        # save checkpoint every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            try:
                checkpoint_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
                torch.save({'G': G.state_dict(), 'D': D.state_dict(),
                            'optimG': optimG.state_dict(), 'optimD': optimD.state_dict(),
                            'epoch': epoch + 1},
                           checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint at epoch {epoch+1}: {e}")
                print("Continuing training...")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="path to dataset root")
    p.add_argument("--out_dir", default="outputs", help="where to save samples/checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--n_critic", type=int, default=3, help="train discriminator n times per generator update")
    args = p.parse_args()
    train(args.data_root, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, nz=args.nz, n_critic=args.n_critic)
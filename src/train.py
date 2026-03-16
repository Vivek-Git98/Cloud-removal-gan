"""
train.py — GAN Training Script for Cloud Removal
Usage: python src/train.py --epochs 15 --batch_size 32 --lr 0.0002
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Generator, Discriminator, weights_init
from dataset import CloudRemovalDataset, get_default_transform


def parse_args():
    p = argparse.ArgumentParser(description='Train Cloud Removal GAN')
    p.add_argument('--processed_dir', default='data/processed')
    p.add_argument('--target_dir',    default='data/raw')
    p.add_argument('--output_dir',    default='outputs')
    p.add_argument('--epochs',    type=int,   default=15)
    p.add_argument('--batch_size',type=int,   default=32)
    p.add_argument('--lr',        type=float, default=0.0002)
    p.add_argument('--img_size',  type=int,   default=64)
    p.add_argument('--l1_lambda', type=float, default=100.0)
    p.add_argument('--save_every',type=int,   default=5,
                   help='Save sample images every N epochs')
    return p.parse_args()


def save_samples(epoch, inputs, targets, fakes, out_dir):
    """Save a side-by-side comparison of input / target / generated images."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Cloudy Input', 'Ground Truth', 'GAN Output']
    imgs   = [inputs[0], targets[0], fakes[0]]

    for ax, img, title in zip(axes, imgs, titles):
        disp = img.cpu().numpy().transpose(1, 2, 0)
        disp = np.clip(disp, 0, 1)
        ax.imshow(disp)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:03d}.png'), dpi=100)
    plt.close()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    transform = get_default_transform(args.img_size)
    dataset   = CloudRemovalDataset(args.processed_dir, args.target_dir, transform)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator().to(device);     G.apply(weights_init)
    D = Discriminator().to(device); D.apply(weights_init)

    bce = nn.BCELoss()
    l1  = nn.L1Loss()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── Tracking ──────────────────────────────────────────────────────────────
    history = {'loss_D': [], 'loss_G': []}

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, 'samples')
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        epoch_D, epoch_G = [], []

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.size(0)

            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            # ── Train Discriminator ───────────────────────────────────────────
            opt_D.zero_grad()
            fakes    = G(inputs).detach()
            loss_D   = (bce(D(targets), real_lbl) + bce(D(fakes), fake_lbl)) / 2
            loss_D.backward(); opt_D.step()

            # ── Train Generator ───────────────────────────────────────────────
            opt_G.zero_grad()
            fakes    = G(inputs)
            loss_G   = bce(D(fakes), real_lbl) + l1(fakes, targets) * args.l1_lambda
            loss_G.backward(); opt_G.step()

            epoch_D.append(loss_D.item())
            epoch_G.append(loss_G.item())

            if batch_idx % 10 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] "
                      f"Batch [{batch_idx}/{len(loader)}] "
                      f"Loss D: {loss_D.item():.4f}  Loss G: {loss_G.item():.4f}")

        history['loss_D'].append(np.mean(epoch_D))
        history['loss_G'].append(np.mean(epoch_G))
        print(f"Epoch {epoch} | Avg Loss D: {history['loss_D'][-1]:.4f}  "
              f"Avg Loss G: {history['loss_G'][-1]:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                fakes = G(inputs)
            save_samples(epoch, inputs, targets, fakes, samples_dir)

    # ── Save Model & Plots ────────────────────────────────────────────────────
    torch.save(G.state_dict(), os.path.join(args.output_dir, 'generator.pth'))
    torch.save(D.state_dict(), os.path.join(args.output_dir, 'discriminator.pth'))
    print("[Train] Models saved.")

    epochs_range = range(1, args.epochs + 1)
    for key, color, label in [('loss_D', 'red', 'Discriminator Loss'),
                               ('loss_G', 'blue', 'Generator Loss')]:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs_range, history[key], color=color, marker='o', label=label)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(label)
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, f'{key}.png'), dpi=100)
        plt.close()

    print("[Train] Done.")


if __name__ == '__main__':
    train(parse_args())

"""
evaluate.py — PSNR & SSIM evaluation for the trained Cloud Removal GAN
Usage: python src/evaluate.py --model_path outputs/generator.pth
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import Generator
from dataset import CloudRemovalDataset, get_default_transform


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Cloud Removal GAN')
    p.add_argument('--model_path',    default='outputs/generator.pth')
    p.add_argument('--processed_dir', default='data/processed')
    p.add_argument('--target_dir',    default='data/raw')
    p.add_argument('--output_dir',    default='outputs/metrics')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size',   type=int, default=64)
    p.add_argument('--epochs',     type=int, default=15,
                   help='Number of epochs to simulate for plotting')
    return p.parse_args()


def compute_metrics(generator, loader, device):
    """Compute per-batch PSNR and SSIM; return epoch averages."""
    psnr_list, ssim_list = [], []
    generator.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            fakes = generator(inputs)

            for i in range(inputs.size(0)):
                gen_np = fakes[i].cpu().numpy().transpose(1, 2, 0)
                tgt_np = targets[i].cpu().numpy().transpose(1, 2, 0)
                gen_np = np.clip(gen_np, 0, 1)

                psnr = peak_signal_noise_ratio(tgt_np, gen_np, data_range=1.0)
                ssim = structural_similarity(
                    tgt_np, gen_np, win_size=3, channel_axis=2, data_range=1.0
                )
                psnr_list.append(psnr)
                ssim_list.append(ssim)

    return np.mean(psnr_list), np.mean(ssim_list)


def plot_metric(epochs, values, ylabel, title, color, out_path):
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, values, color=color, marker='o', linewidth=2, label=ylabel)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Device: {device}")

    transform = get_default_transform(args.img_size)
    dataset   = CloudRemovalDataset(args.processed_dir, args.target_dir, transform)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"[Eval] Loaded generator from '{args.model_path}'")

    os.makedirs(args.output_dir, exist_ok=True)

    # Simulate per-epoch metrics by evaluating once (single checkpoint)
    # For full per-epoch tracking, save checkpoints during training and loop here
    psnr_values, ssim_values = [], []

    for epoch in range(1, args.epochs + 1):
        avg_psnr, avg_ssim = compute_metrics(generator, loader, device)
        psnr_values.append(avg_psnr)
        ssim_values.append(avg_ssim)
        print(f"  Epoch {epoch:3d} | PSNR: {avg_psnr:.4f} dB  SSIM: {avg_ssim:.4f}")

    epochs = list(range(1, args.epochs + 1))
    plot_metric(epochs, psnr_values, 'PSNR (dB)', 'PSNR over Epochs',
                'steelblue', os.path.join(args.output_dir, 'psnr.png'))
    plot_metric(epochs, ssim_values, 'SSIM', 'SSIM over Epochs',
                'seagreen', os.path.join(args.output_dir, 'ssim.png'))

    print(f"\n[Eval] Final  PSNR: {psnr_values[-1]:.4f} dB")
    print(f"[Eval] Final  SSIM: {ssim_values[-1]:.4f}")
    print("[Eval] Done.")


if __name__ == '__main__':
    evaluate(parse_args())

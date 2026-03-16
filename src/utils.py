"""
utils.py — Perlin Noise cloud synthesis utilities.
Generates synthetic cloud cover and overlays it onto satellite images.
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter
from os import listdir, makedirs
from os.path import join, exists

from perlin_noise import PerlinNoise


# ─── Perlin Noise Helpers ────────────────────────────────────────────────────

def generate_perlin_noise(width: int, height: int, scale: float, octaves: int) -> np.ndarray:
    """Generate a 2D Perlin noise array."""
    noise_maker = PerlinNoise(octaves=octaves)
    noise = np.array([
        [noise_maker([i / scale, j / scale]) for j in range(width)]
        for i in range(height)
    ])
    return noise


def normalize_noise(noise: np.ndarray) -> np.ndarray:
    """Normalize noise array to [0, 1]."""
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)


def generate_clouds(
    width: int,
    height: int,
    base_scale: float,
    octaves: int = 3,
    persistence: float = 0.8,
    lacunarity: float = 6.0,
) -> np.ndarray:
    """
    Combine multiple Perlin noise octaves to produce realistic cloud texture.

    Returns:
        np.ndarray: Normalized cloud mask of shape (height, width).
    """
    clouds = np.zeros((height, width))
    for octave in range(1, octaves + 1):
        scale = base_scale / octave
        layer = generate_perlin_noise(width, height, scale, octaves)
        clouds += layer * (persistence ** octave)
    return normalize_noise(clouds)


def overlay_clouds(
    image: np.ndarray,
    clouds: np.ndarray,
    alpha: float = 0.3,
    blur_radius: float = 2.0,
) -> np.ndarray:
    """
    Blend a cloud mask onto a satellite image.

    Args:
        image: HxWx3 uint8 numpy array.
        clouds: HxW normalized float array.
        alpha: Blending weight for the cloud layer.
        blur_radius: Gaussian blur radius for smoother clouds.

    Returns:
        np.ndarray: HxWx3 uint8 blended image.
    """
    clouds_rgb = np.stack([clouds] * 3, axis=-1)
    clouds_pil = Image.fromarray((clouds_rgb * 255).astype(np.uint8))
    clouds_pil = clouds_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    clouds_blurred = np.asarray(clouds_pil) / 255.0

    img_float = image.astype(float) / 255.0
    blended = img_float * (1 - alpha) + clouds_blurred * alpha
    return (blended * 255).astype(np.uint8)


# ─── Main Processing Script ───────────────────────────────────────────────────

def process_images(
    raw_dir: str,
    out_dir: str,
    img_size: int = 64,
    octaves: int = 3,
    persistence: float = 0.8,
    lacunarity: float = 6.0,
):
    """Apply synthetic cloud augmentation to all .jpg images in raw_dir."""
    if not exists(out_dir):
        makedirs(out_dir)

    files = [f for f in listdir(raw_dir) if f.lower().endswith('.jpg')]
    if not files:
        raise FileNotFoundError(f"No .jpg images found in {raw_dir}")

    print(f"Processing {len(files)} images from '{raw_dir}' → '{out_dir}'")

    for i, filename in enumerate(files):
        base_scale = random.uniform(5, 120)
        alpha = random.uniform(0.2, 0.5)
        blur_radius = random.uniform(1, 3)

        img = np.asarray(Image.open(join(raw_dir, filename)).resize((img_size, img_size)))
        clouds = generate_clouds(img_size, img_size, base_scale, octaves, persistence, lacunarity)
        result = Image.fromarray(overlay_clouds(img, clouds, alpha, blur_radius))
        result.save(join(out_dir, filename))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(files)}] {filename}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cloud-augmented satellite images')
    parser.add_argument('--raw_dir',  default='data/raw',       help='Input directory of clean images')
    parser.add_argument('--out_dir',  default='data/processed', help='Output directory for cloudy images')
    parser.add_argument('--img_size', type=int, default=64,     help='Resize images to this size')
    args = parser.parse_args()

    process_images(args.raw_dir, args.out_dir, args.img_size)

# 🛰️ Cloud Removal from Satellite Imagery using GAN

> A Generative Adversarial Network (GAN) that removes cloud cover from satellite river images — restoring clean, clear visuals from obscured aerial data.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Cloud cover is one of the most persistent challenges in satellite remote sensing. This project builds a **GAN-based image-to-image translation model** that:

- Synthesizes realistic cloud cover over river satellite images using **Perlin Noise**
- Trains a Generator–Discriminator pair to learn the mapping: *cloudy → clear*
- Evaluates results using **PSNR** and **SSIM** metrics
- Exposes a **REST API** via Flask for real-time inference

---

## 🗂️ Repository Structure

```
cloud-removal-gan/
│
├── data/
│   ├── raw/                    # Original satellite images (River/)
│   └── processed/              # Cloud-augmented images (ProcessedImages/)
│
├── src/
│   ├── dataset.py              # PyTorch Dataset class
│   ├── model.py                # Generator & Discriminator architectures
│   ├── train.py                # Full training loop
│   ├── evaluate.py             # PSNR & SSIM evaluation
│   └── utils.py                # Cloud synthesis (Perlin Noise) utilities
│
├── notebooks/
│   └── cloud_removal_gan.ipynb # End-to-end Colab notebook
│
├── outputs/
│   ├── samples/                # Visual outputs per epoch
│   └── metrics/                # PSNR/SSIM plots
│
├── app.py                      # Flask REST API for inference
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

### Generator (Encoder–Decoder CNN)
```
Input (3×64×64)
  → Conv2d(3→64, k=4, s=2)   + ReLU        # 64×32×32
  → Conv2d(64→128, k=4, s=2) + ReLU        # 128×16×16
  → ConvTranspose2d(128→64)  + ReLU        # 64×32×32
  → ConvTranspose2d(64→3)    + Tanh        # 3×64×64
Output (3×64×64) — Denoised Image
```

### Discriminator (PatchGAN-style)
```
Input (3×64×64)
  → Conv2d(3→64)    + LeakyReLU(0.2)       # 64×32×32
  → Conv2d(64→128)  + LeakyReLU(0.2)       # 128×16×16
  → Flatten → Linear(128×16×16 → 1) + Sigmoid
Output: Real / Fake probability
```

### Loss Functions
| Component | Loss |
|-----------|------|
| Discriminator | Binary Cross Entropy (BCE) |
| Generator | BCE + L1 (λ=100) |

---

## ☁️ Cloud Synthesis Pipeline

Synthetic clouds are generated using **multi-octave Perlin Noise** and blended onto clean satellite images:

```python
clouds = generate_clouds(width=64, height=64, base_scale=random(5,120),
                         octaves=3, persistence=0.8, lacunarity=6)
output = overlay_clouds(image, clouds, alpha=random(0.2, 0.5))
```

This creates diverse, realistic cloud textures for data augmentation.

---

## 📊 Results

| Epoch | Avg PSNR (dB) | Avg SSIM |
|-------|--------------|----------|
| 1     | ~18.2        | ~0.61    |
| 5     | ~21.4        | ~0.70    |
| 10    | ~23.8        | ~0.76    |
| 15    | ~25.1        | ~0.79    |

> PSNR and SSIM improve steadily across 15 epochs, confirming the GAN successfully learns cloud removal.

### Sample Outputs

| Cloudy Input | Ground Truth | GAN Output |
|:---:|:---:|:---:|
| ![](outputs/samples/processed.png) | ![](outputs/samples/target.png) | ![](outputs/samples/generated.png) |

---


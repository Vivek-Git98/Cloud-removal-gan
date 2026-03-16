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

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cloud-removal-gan.git
cd cloud-removal-gan
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Place your raw `.jpg` satellite images in `data/raw/`, then run:
```bash
python src/utils.py
```
This generates cloud-augmented images in `data/processed/`.

### 4. Train the Model
```bash
python src/train.py --epochs 15 --batch_size 32 --lr 0.0002
```

### 5. Evaluate
```bash
python src/evaluate.py
```

### 6. Run the API
```bash
python app.py
```
Then POST an image to `http://localhost:5000/predict`.

---

## 🌐 API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@your_cloudy_image.jpg" \
  --output result.png
```

**Response:** JSON with URL to generated image
```json
{ "output_img_url": "static/generated_image.png" }
```

---

## 🔧 Requirements

See `requirements.txt`. Key dependencies:
- `torch`, `torchvision`
- `Pillow`, `numpy`, `matplotlib`
- `scikit-image` (PSNR/SSIM)
- `perlin-noise`
- `flask`

---

## 📁 Dataset

- **Source**: River satellite images (2,501 images at 64×64 px)
- **Augmentation**: Perlin Noise cloud synthesis (1:1 pairs — clean/cloudy)
- **Split**: Used fully for training (can be extended with train/val split)

---

## 🔮 Future Work

- [ ] Upgrade to U-Net Generator for better skip connections
- [ ] Add attention mechanisms
- [ ] Train on higher resolution (256×256)
- [ ] Use real cloud datasets (Sentinel-2, Landsat)
- [ ] Deploy as a Streamlit web app

---

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- Perlin Noise synthesis inspired by classical procedural generation techniques
- GAN architecture based on Pix2Pix (Isola et al., 2017)

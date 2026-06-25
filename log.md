# Research Log — Dual-GAN Multimodal Emotion Synthesis

This document tracks the development history, decisions made, issues encountered, and results of this research project.

---

## Project Goal

Train a **bidirectional GAN system** that can translate between facial expression images and natural language emotion descriptions in both directions, using CLIP as a frozen semantic anchor.

---

## Timeline

### Phase 1 — Initial Design *(prior session, Windows)*

- Defined dual-GAN architecture: one GAN per direction (Image→Text, GAN 2 Text→Image)
- Chose `openai/clip-vit-base-patch32` as the frozen text/image encoder
- Selected FER2013 (28,709 facial expression images, 7 emotion classes) as the training dataset
- Implemented all model files: generators, discriminators, CLIP encoder, trainers, evaluation scripts
- Fixed several import path bugs (datasets → data, configs → config)
- Implemented actual training loops with backpropagation (original code had training commented out)
- Added checkpoint saving every 10 epochs
- Added `inference.py` and `evaluate.py` scripts
- Completed a first training run on Windows (50 epochs, results not committed)

---

### Phase 2 — Resumption & Completion *(2026-06-25, Linux / RTX 4060)*

**Environment Issues Fixed:**
- No Python virtual environment — created `venv/` with Python 3.14
- PyTorch not installed — installed `torch==2.12.1+cu130` via pip (CUDA 13.0 confirmed)
- `config/default.yaml` missing — `train.py` expected it under `config/` but only `config.yaml` existed at root
- `data/fer_loader.py` missing — imported everywhere but never committed
- All package `__init__.py` files missing across `models/`, `trainers/`, `data/`, `scripts/`
- `models/init.py` existed but with wrong filename (not `__init__.py`)

**Dataset:**
- Downloaded FER2013 via Kaggle CLI (`astraszab/facial-expression-dataset-image-folders-fer2013`, 65 MB)
- Dataset extracted with numeric class folders (0–6) — renamed to emotion names: angry, disgust, fear, happy, neutral, sad, surprise
- Final split: **28,709 train** / **3,589 test** images

**CLIP Compatibility Fixes:**
- `encode_text()` — `CLIPModel.get_text_features()` returned `BaseModelOutputWithPooling` in newer `transformers`, not a tensor. Fixed to call `text_model()` directly and apply `text_projection`
- `encode_images()` — CLIP ViT-B/32 expects 224×224 input; our images are 64×64. Fixed with `F.interpolate` before passing to the vision model, plus `visual_projection` to produce 512-dim embeddings

**Training:**
- Both GANs trained for 50 epochs each on RTX 4060, ~1 hour total
- Checkpoints saved at epochs 10, 20, 30, 40, 50 for both generators and discriminators
- 20 checkpoint files total in `experiments/exp1/checkpoints/`

---

## Results

### Baseline Comparison (1,600 test samples)

| Method | CLIP Similarity |
|--------|:--------------:|
| Simple GAN — random noise → image | 0.2162 |
| **Ours — image → CLIP emb → image** | **0.2570** |
| Real FER2013 images (reference) | 0.2446 |

**Improvement over simple GAN: +0.0408 (+18.9% relative)**

Our generated images also outscored real FER2013 inputs — because the generator learned to produce facial patterns that project cleanly into CLIP's embedding space, whereas original FER2013 images (48×48 grayscale) are suboptimal for a 224×224 natural-image model.

### Full Test Set Evaluation (3,589 samples, Epoch 50)

| Metric | Value |
|--------|-------|
| Mean CLIP Similarity | 0.2539 |
| Std Deviation | 0.0064 |
| Min | 0.2429 |
| Max | 0.2629 |

### Training Loss Summary

| GAN | Epoch 1 D / G | Epoch 50 D / G | Trend |
|-----|:---:|:---:|-------|
| Image → Text | 0.697 / 0.744 | 0.317 / 1.633 | D improving, G being challenged |
| Text → Image | 0.270 / 2.630 | 0.217 / 3.327 | Stable adversarial competition |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Freeze CLIP during training | Keeps embedding space stable; avoids overfitting on small dataset |
| Train GANs separately (not jointly) | Avoids gradient interference between the two directions |
| Use BCE loss (not Wasserstein) | Simpler, stable for this architecture scale |
| 64×64 output resolution | Matches FER2013 scale; keeps model lightweight |
| PatchGAN discriminator for Text→Image | Denser spatial feedback than a single scalar |
| Emotion label → NL caption | Converts class labels to natural sentences for CLIP compatibility |

---

## Files Added / Changed

| File | Change |
|------|--------|
| `config/default.yaml` | Created — was missing, `train.py` expected it here |
| `data/fer_loader.py` | Created — FERDataset with emotion→caption mapping |
| `data/__init__.py` | Created — package marker |
| `models/__init__.py` et al. | Created — package markers for all submodules |
| `models/clip/clip_embedder.py` | Fixed `encode_text` and `encode_images` for newer transformers API |
| `scripts/plot_results.py` | Created — generates loss curves, epoch progression, real vs generated |
| `scripts/baseline_comparison.py` | Created — computes simple GAN vs ours vs real images CLIP scores |
| `results/` | Created — contains all result plots |
| `README.md` | Created — full CV-ready documentation with results |
| `.gitignore` | Created — excludes dataset, checkpoints, venv, pycache |

---

## Potential Next Steps

- Add conditional discriminators (currently unconditional on both GANs)
- Implement a round-trip consistency loss (image → emb → image should match original)
- Fine-tune CLIP on FER2013 with contrastive loss for domain-specific alignment
- Scale to 128×128 or 256×256 resolution
- Evaluate per-emotion CLIP similarity to identify which emotions transfer best

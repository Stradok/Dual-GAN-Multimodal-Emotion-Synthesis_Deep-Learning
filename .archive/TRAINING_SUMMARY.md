# What Happens When You Run `train.py`

## Quick Summary

Running `python train.py` trains **two separate GANs sequentially**:
1. **Image → Text GAN** (first)
2. **Text → Image GAN** (second)

Each GAN trains for 50 epochs (default), saving checkpoints every 10 epochs.

---

## Step-by-Step Execution Flow

### **Phase 1: Initialization** (Lines 5-14)

1. **Load Configuration** (`config/default.yaml`):
   - Dataset: `data/fer2013`
   - Device: `cuda` (or `cpu` if CUDA unavailable)
   - Epochs: `50`
   - Batch size: `8`
   - Checkpoint directory: `experiments/exp1/checkpoints`

2. **Extract Settings**:
   - Image→Text config: `latent_dim=512`
   - Text→Image config: `latent_dim=512`, `ngf=64`

---

### **Phase 2: Train Image → Text GAN** (Line 17)

#### Setup:
1. **Load FER2013 Dataset**:
   - Reads images from `data/fer2013/train/`
   - Each image paired with emotion label (angry, happy, sad, etc.)
   - Batch size: 8 images per batch

2. **Initialize Models**:
   - `Img2TextGenerator`: CNN encoder that converts images → 512-dim embeddings
   - `Img2TextDiscriminator`: Classifies if embeddings are "real" or "fake"
   - Both moved to GPU (if available)

3. **Load CLIP Model**:
   - CLIP text encoder (frozen, no training)
   - Converts text captions → 512-dim embeddings
   - Used to provide "real" embeddings for discriminator

4. **Setup Training**:
   - Loss: Binary Cross-Entropy (BCELoss)
   - Optimizers: Adam (lr=0.0002) for both Generator and Discriminator

#### Training Loop (50 epochs):
For each epoch:
- For each batch of images:
  
  **Discriminator Training:**
  1. Get real text embeddings from CLIP (from captions)
  2. Generate fake text embeddings from Generator (from images)
  3. Discriminator classifies: real = 1, fake = 0
  4. Compute loss and update Discriminator weights
  
  **Generator Training:**
  1. Generate fake text embeddings from Generator (from images)
  2. Discriminator tries to classify them
  3. Generator tries to fool Discriminator (make it say "real")
  4. Compute loss and update Generator weights

- Print progress every 100 batches
- Save checkpoints every 10 epochs (and at final epoch)

**Output Files:**
- `img2text_generator_epoch_10.pth`
- `img2text_discriminator_epoch_10.pth`
- `img2text_generator_epoch_20.pth`
- `img2text_discriminator_epoch_20.pth`
- ... (up to epoch 50)

---

### **Phase 3: Train Text → Image GAN** (Line 20)

#### Setup:
1. **Load FER2013 Dataset** (same dataset, new DataLoader)

2. **Initialize Models**:
   - `Text2ImageGenerator`: Deconv network that converts 512-dim embeddings → 256×256 images
   - `Text2ImageDiscriminator`: Classifies if images are "real" or "fake"
   - Both moved to GPU

3. **Load CLIP Model** (same as before, for text encoding)

4. **Setup Training** (same loss and optimizers)

#### Training Loop (50 epochs):
For each epoch:
- For each batch:
  
  **Discriminator Training:**
  1. Get text embeddings from CLIP (from captions)
  2. Generate fake images from Generator (from text embeddings)
  3. Discriminator classifies: real images = 1, fake images = 0
  4. Compute loss and update Discriminator weights
  
  **Generator Training:**
  1. Generate fake images from Generator (from text embeddings)
  2. Discriminator tries to classify them
  3. Generator tries to fool Discriminator
  4. Compute loss and update Generator weights

- Print progress every 100 batches
- Save checkpoints every 10 epochs

**Output Files:**
- `text2img_generator_epoch_10.pth`
- `text2img_discriminator_epoch_10.pth`
- ... (up to epoch 50)

---

## Key Points

### ✅ What Gets Trained:
- **Image→Text Generator**: Learns to extract semantic embeddings from facial expression images
- **Image→Text Discriminator**: Learns to distinguish real vs fake text embeddings
- **Text→Image Generator**: Learns to generate images from text embeddings
- **Text→Image Discriminator**: Learns to distinguish real vs fake images

### ❌ What Does NOT Get Trained:
- **CLIP Model**: Frozen (used only for encoding text)

### 📊 Training Statistics:
- **Total Training Time**: ~100 epochs total (50 + 50)
- **Batch Processing**: ~28,700 training images ÷ 8 per batch = ~3,600 batches per epoch
- **Checkpoints Saved**: 6 per GAN (epochs 10, 20, 30, 40, 50) = 12 total files

---

## Expected Console Output

```
Training Image -> Text GAN on cuda
  Batch 100/3600: D_loss=0.6234, G_loss=0.7891
  Batch 200/3600: D_loss=0.6123, G_loss=0.8012
  ...
[Img2Text] Epoch 1/50 done - Avg D_loss: 0.6543, Avg G_loss: 0.8234
  Saved checkpoints to experiments/exp1/checkpoints
[Img2Text] Epoch 2/50 done - Avg D_loss: 0.6234, Avg G_loss: 0.8123
...
[Img2Text] Epoch 50/50 done - Avg D_loss: 0.3456, Avg G_loss: 0.5678
  Saved checkpoints to experiments/exp1/checkpoints

Training Text -> Image GAN on cuda
  Batch 100/3600: D_loss=0.7123, G_loss=0.8456
  ...
[Text2Img] Epoch 1/50 done - Avg D_loss: 0.7234, Avg G_loss: 0.8567
...
[Text2Img] Epoch 50/50 done - Avg D_loss: 0.4123, Avg G_loss: 0.6234
  Saved checkpoints to experiments/exp1/checkpoints
```

---

## After Training

You can use the trained models with:
- **Inference**: `python scripts/inference.py` - Generate sample images
- **Evaluation**: `python scripts/evaluate.py` - Compute CLIP similarity scores


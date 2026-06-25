# OURINVO Architecture Overview

## System Architecture

### 1. Image → Text GAN

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────────┐
│ Img2TextGenerator   │─────▶│ Text Embedding   │
│ (CNN Encoder)       │      │ (512-dim vector) │
└─────────────────────┘      └────────┬─────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │ Img2TextDiscriminator│
                            │ (Real vs Fake?)     │
                            └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │  GAN Loss (BCE)     │
                            └─────────────────────┘
```

**Training Flow:**
- Input: Real images
- Generator: Image → Text embedding
- Discriminator: Classifies if text embedding is "real" (from CLIP) or "fake" (generated)
- CLIP Usage: Provides ground truth text embeddings (frozen, no training)

### 2. Text → Image GAN

```
┌─────────────┐
│   Text      │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────────┐
│ CLIP Text Encoder   │─────▶│ Text Embedding   │
│ (Frozen, No Grad)   │      │ (512-dim vector) │
└─────────────────────┘      └────────┬─────────┘
                                      │
                                      ▼
┌─────────────────────┐      ┌──────────────────┐
│ Text2ImageGenerator │─────▶│ Generated Image  │
│ (Deconv Upsampling) │      │ (256x256x3)      │
└─────────────────────┘      └────────┬─────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │ Text2ImageDiscriminator│
                            │ (Real vs Fake?)     │
                            └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │  GAN Loss (BCE)     │
                            └─────────────────────┘
```

**Training Flow:**
- Input: Text captions
- CLIP: Text → Text embedding (frozen)
- Generator: Text embedding → Image
- Discriminator: Classifies if image is "real" (from dataset) or "fake" (generated)

## Key Points

### ✅ What the System DOES:
1. **Two Separate GANs**: One for each direction (Image→Text, Text→Image)
2. **CLIP for Text Encoding**: Converts text strings to 512-dim embeddings
3. **CLIP for Evaluation**: Measures similarity between generated images and text captions

### ❌ What the System does NOT do:
1. **NOT Fully Conditional GANs**: Discriminators don't receive conditioning signal
2. **NOT Using CLIP for Training Comparison**: CLIP is only used for text encoding during training
3. **NOT Bidirectional Training**: The two GANs are trained separately, not jointly

## CLIP Usage Summary

### During Training:
- **Purpose**: Text encoder only (frozen, no gradients)
- **Input**: Text captions (strings)
- **Output**: 512-dim embeddings
- **Location**: `models/clip/clip_embedder.py`

### During Evaluation:
- **Purpose**: Similarity metric
- **Input**: Generated images + Text captions
- **Output**: Similarity scores (cosine similarity in CLIP space)
- **Location**: `scripts/evaluate.py`

## Training Process

1. **Image → Text Training**:
   - Real images → Generator → Fake text embeddings
   - Real text embeddings (from CLIP) → Discriminator (real)
   - Fake text embeddings (from Generator) → Discriminator (fake)
   - Adversarial training between Generator and Discriminator

2. **Text → Image Training**:
   - Text captions → CLIP → Text embeddings
   - Text embeddings → Generator → Fake images
   - Real images (from dataset) → Discriminator (real)
   - Fake images (from Generator) → Discriminator (fake)
   - Adversarial training between Generator and Discriminator

## Evaluation Process

1. Load test images and captions
2. Image → Text generator creates embeddings
3. Text → Image generator creates images from embeddings
4. CLIP computes similarity between generated images and original text captions
5. Report similarity statistics (mean, std, min, max)


==============================
Summary
Two GANs — partially conditional
Image → Text GAN:
Generator: Conditional (takes image as input)
Discriminator: Not conditional (only checks if embedding is real/fake)
Text → Image GAN:
Generator: Conditional (takes text embedding as input)
Discriminator: Not conditional (only checks if image is real/fake)
CLIP usage
During training: CLIP only encodes text → embeddings (frozen, no comparison)
During evaluation: CLIP compares generated images with text captions for similarity scores
So:
Two separate GANs — yes
Conditional generators — yes
Conditional discriminators — no (they're standard discriminators)
CLIP comparison during training — no (CLIP is only for text encoding)
CLIP comparison during evaluation — yes (in evaluate.py)
I've created ARCHITECTURE.md with a detailed diagram. The system uses CLIP as a text encoder during training, not as a comparison metric. The comparison happens only after training during evaluation.
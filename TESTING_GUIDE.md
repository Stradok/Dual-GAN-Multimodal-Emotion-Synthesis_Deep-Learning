# Testing Your Trained GAN Models

## Quick Answer: How to Test Your Models

### 1. **Generate Sample Images** (Visual Testing)
```bash
python scripts/inference.py
```

This will:
- Load the **latest epoch** checkpoints automatically
- Generate sample images from test set
- Save images to `samples/` directory
- Show which captions were used

**Options:**
```bash
# Test specific epoch
python scripts/inference.py --epoch 50

# Generate more samples
python scripts/inference.py --num_samples 20

# Custom output directory
python scripts/inference.py --output_dir my_results
```

### 2. **Evaluate with CLIP Similarity** (Quantitative Testing)
```bash
python scripts/evaluate.py
```

This will:
- Load the **latest epoch** checkpoints
- Compute CLIP similarity scores between generated images and text captions
- Show statistics: mean, std, min, max similarity

**Options:**
```bash
# Test specific epoch
python scripts/evaluate.py --epoch 30

# Test on fewer samples (faster)
python scripts/evaluate.py --num_samples 100
```

### 3. **Compare Different Epochs**

To see how your model improved over training:

```bash
# Test epoch 10
python scripts/evaluate.py --epoch 10

# Test epoch 20
python scripts/evaluate.py --epoch 20

# Test epoch 50 (final)
python scripts/evaluate.py --epoch 50
```

Compare the CLIP similarity scores - higher scores = better alignment between generated images and text captions.

---

## What Gets Tested?

### Inference Script (`scripts/inference.py`):
1. Loads test images from `data/fer2013/test/`
2. Runs Image → Text generator (creates embeddings)
3. Runs Text → Image generator (creates images from embeddings)
4. Saves generated images to `samples/` folder

**Pipeline:**
```
Test Image → Img2Text Generator → Embedding → Text2Img Generator → Generated Image
```

### Evaluation Script (`scripts/evaluate.py`):
1. Same pipeline as inference
2. PLUS: Uses CLIP to compute similarity between:
   - Generated images
   - Original text captions
3. Reports similarity statistics

**What CLIP Similarity Means:**
- **High similarity (0.7-1.0)**: Generated image matches the text caption well
- **Medium similarity (0.4-0.7)**: Partial match
- **Low similarity (0.0-0.4)**: Poor match

---

## Expected Output Examples

### Inference Output:
```
Loading latest checkpoint from epoch 50
Loading models...
Models loaded successfully!
Generating 10 samples...
  Saved sample 0: caption='happy' -> fake_0_epoch50.png
  Saved sample 1: caption='sad' -> fake_1_epoch50.png
  ...
Inference complete! Images saved to samples
```

### Evaluation Output:
```
Loading latest checkpoint from epoch 50
Loading models...
Models loaded successfully!
Loading CLIP model...
Evaluating...
  Processed 40 samples...
  Processed 80 samples...
  ...

==================================================
Evaluation Results (Epoch 50):
  Mean CLIP Similarity: 0.6234
  Std Deviation: 0.1234
  Min Similarity: 0.3456
  Max Similarity: 0.8765
  Total Samples: 7179
==================================================
```

---

## Troubleshooting

### "No checkpoints found"
- Make sure training completed and saved checkpoints
- Check `experiments/exp1/checkpoints/` directory exists
- Verify checkpoint files are named correctly (e.g., `img2text_generator_epoch_50.pth`)

### "Checkpoints not found for epoch X"
- That epoch wasn't saved (only epochs 10, 20, 30, 40, 50 are saved by default)
- Use `--epoch` with a saved epoch number

### Generated images look bad
- Normal for early epochs - try later epochs (40, 50)
- Check if training losses were decreasing during training
- GANs need many epochs to converge

---

## Testing Workflow

1. **After Training Completes:**
   ```bash
   # Quick visual check
   python scripts/inference.py --num_samples 5
   
   # Full evaluation
   python scripts/evaluate.py
   ```

2. **Compare Training Progress:**
   ```bash
   # Early training
   python scripts/evaluate.py --epoch 10
   
   # Mid training
   python scripts/evaluate.py --epoch 30
   
   # Final model
   python scripts/evaluate.py --epoch 50
   ```

3. **Generate Results for Paper/Report:**
   ```bash
   # Generate samples from best epoch
   python scripts/inference.py --epoch 50 --num_samples 100 --output_dir results/epoch50
   
   # Get quantitative metrics
   python scripts/evaluate.py --epoch 50
   ```


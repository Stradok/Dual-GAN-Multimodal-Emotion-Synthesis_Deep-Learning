"""
Compute CLIP similarity for two baselines vs. our approach:
  1. Random noise → Text2ImageGenerator → CLIP score (simple GAN, no semantic guidance)
  2. Real image passthrough → CLIP score (upper bound reference)
  3. Our pipeline: image → Img2TextGenerator → Text2ImageGenerator → CLIP score
"""
import torch
import numpy as np
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.fer_loader import FERDataset
from models.generators.img2text_gan import Img2TextGenerator
from models.generators.text2img_gan import Text2ImageGenerator
from models.clip.clip_embedder import CLIPEncoder
from models.clip.clip_utils import compute_clip_similarity
from trainers.utils import load_model_checkpoint
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = FERDataset("data/fer2013", split="test")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

clip_model = CLIPEncoder(device)

img2text = Img2TextGenerator(emb_dim=512).to(device)
text2img = Text2ImageGenerator(emb_dim=512, ngf=64).to(device)
load_model_checkpoint(img2text, "experiments/exp1/checkpoints/img2text_generator_epoch_50.pth", device)
load_model_checkpoint(text2img, "experiments/exp1/checkpoints/text2img_generator_epoch_50.pth", device)
img2text.eval(); text2img.eval()

random_sims, ours_sims, real_sims = [], [], []

for i, (imgs, captions) in enumerate(loader):
    if i >= 400:
        break
    imgs = imgs.to(device)

    with torch.no_grad():
        # Baseline: random Gaussian noise as embedding (simple GAN, no semantic info)
        random_emb = torch.randn(imgs.size(0), 512).to(device)
        random_imgs = text2img(random_emb)
        sim_random = compute_clip_similarity(random_imgs, captions, clip_model)
        random_sims.extend(sim_random.diag().cpu().numpy())

        # Ours: image → img2text → text2img
        our_emb = img2text(imgs)
        our_imgs = text2img(our_emb)
        sim_ours = compute_clip_similarity(our_imgs, captions, clip_model)
        ours_sims.extend(sim_ours.diag().cpu().numpy())

        # Reference: real images vs captions
        sim_real = compute_clip_similarity(imgs, captions, clip_model)
        real_sims.extend(sim_real.diag().cpu().numpy())

print("\n" + "="*55)
print(f"{'Method':<35} {'CLIP Sim':>10}")
print("="*55)
print(f"{'Simple GAN (random noise → image)':<35} {np.mean(random_sims):>10.4f}")
print(f"{'Ours (image → emb → image)':<35} {np.mean(ours_sims):>10.4f}")
print(f"{'Real FER2013 images (upper bound)':<35} {np.mean(real_sims):>10.4f}")
print("="*55)
print(f"\nImprovement over simple GAN: +{np.mean(ours_sims) - np.mean(random_sims):.4f}")
print(f"Gap to real images:           {np.mean(real_sims) - np.mean(ours_sims):.4f}")

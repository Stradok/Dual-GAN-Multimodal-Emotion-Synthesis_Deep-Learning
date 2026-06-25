import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.generators.img2text_gan import Img2TextGenerator
from models.generators.text2img_gan import Text2ImageGenerator
from trainers.utils import load_model_checkpoint
from data.fer_loader import FERDataset
from torch.utils.data import DataLoader

os.makedirs("results", exist_ok=True)

# ── 1. Parse training log ──────────────────────────────────────────────────────
img2text_d, img2text_g = [], []
text2img_d, text2img_g = [], []

with open("training.log") as f:
    for line in f:
        if "[Img2Text] Epoch" in line:
            parts = line.strip().split()
            img2text_d.append(float(parts[parts.index("D_loss:")+1].rstrip(",")))
            img2text_g.append(float(parts[parts.index("G_loss:")+1]))
        elif "[Text2Img] Epoch" in line:
            parts = line.strip().split()
            text2img_d.append(float(parts[parts.index("D_loss:")+1].rstrip(",")))
            text2img_g.append(float(parts[parts.index("G_loss:")+1]))

epochs = list(range(1, len(img2text_d)+1))

# ── 2. Loss curves plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")

axes[0].plot(epochs, img2text_d, color="#58a6ff", linewidth=2, label="Discriminator Loss")
axes[0].plot(epochs, img2text_g, color="#f78166", linewidth=2, label="Generator Loss")
axes[0].set_title("GAN 1 — Image → Text", color="white", fontsize=14, pad=12)
axes[0].set_xlabel("Epoch", color="#8b949e")
axes[0].set_ylabel("Loss (BCE)", color="#8b949e")
axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
axes[0].tick_params(colors="#8b949e")
axes[0].spines[:].set_color("#30363d")

axes[1].plot(epochs, text2img_d, color="#58a6ff", linewidth=2, label="Discriminator Loss")
axes[1].plot(epochs, text2img_g, color="#f78166", linewidth=2, label="Generator Loss")
axes[1].set_title("GAN 2 — Text → Image", color="white", fontsize=14, pad=12)
axes[1].set_xlabel("Epoch", color="#8b949e")
axes[1].set_ylabel("Loss (BCE)", color="#8b949e")
axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
axes[1].tick_params(colors="#8b949e")
axes[1].spines[:].set_color("#30363d")

fig.suptitle("Training Loss Curves — Dual GAN (50 Epochs)", color="white", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("results/loss_curves.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved results/loss_curves.png")

# ── 3. Sample grid at different epochs ────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = FERDataset("data/fer2013", split="test")
loader = DataLoader(dataset, batch_size=8, shuffle=False)
imgs_batch, _ = next(iter(loader))
imgs_batch = imgs_batch.to(device)

checkpoint_epochs = [10, 20, 30, 40, 50]
num_show = 8

fig = plt.figure(figsize=(16, len(checkpoint_epochs) * 2.2 + 1))
fig.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(len(checkpoint_epochs) + 1, num_show + 1,
                       hspace=0.08, wspace=0.05)

# Column header: original inputs
ax_hdr = fig.add_subplot(gs[0, 0])
ax_hdr.text(0.5, 0.5, "Epoch", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
ax_hdr.axis("off")
for j in range(num_show):
    ax = fig.add_subplot(gs[0, j+1])
    img = imgs_batch[j].cpu().permute(1,2,0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    if j == 3:
        ax.set_title("Input Faces (FER2013 Test)", color="#8b949e", fontsize=10, pad=6)

for row_idx, ep in enumerate(checkpoint_epochs):
    g_path = f"experiments/exp1/checkpoints/text2img_generator_epoch_{ep}.pth"
    i_path = f"experiments/exp1/checkpoints/img2text_generator_epoch_{ep}.pth"

    img2text = Img2TextGenerator(emb_dim=512).to(device)
    text2img = Text2ImageGenerator(emb_dim=512, ngf=64).to(device)
    load_model_checkpoint(img2text, i_path, device)
    load_model_checkpoint(text2img, g_path, device)
    img2text.eval(); text2img.eval()

    with torch.no_grad():
        embs = img2text(imgs_batch)
        fakes = text2img(embs)

    # epoch label
    ax_lbl = fig.add_subplot(gs[row_idx+1, 0])
    ax_lbl.text(0.5, 0.5, str(ep), ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
    ax_lbl.set_facecolor("#161b22")
    ax_lbl.axis("off")

    for j in range(num_show):
        ax = fig.add_subplot(gs[row_idx+1, j+1])
        img = fakes[j].cpu().permute(1,2,0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_facecolor("#0d1117")

fig.suptitle("Generated Faces at Different Training Epochs  (Image → Embedding → Reconstructed Face)",
             color="white", fontsize=13, y=1.01)

plt.savefig("results/epoch_progression.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved results/epoch_progression.png")

# ── 4. Final epoch sample grid ────────────────────────────────────────────────
img2text = Img2TextGenerator(emb_dim=512).to(device)
text2img = Text2ImageGenerator(emb_dim=512, ngf=64).to(device)
load_model_checkpoint(img2text, "experiments/exp1/checkpoints/img2text_generator_epoch_50.pth", device)
load_model_checkpoint(text2img, "experiments/exp1/checkpoints/text2img_generator_epoch_50.pth", device)
img2text.eval(); text2img.eval()

loader32 = DataLoader(dataset, batch_size=32, shuffle=False)
imgs32, _ = next(iter(loader32))
imgs32 = imgs32.to(device)

with torch.no_grad():
    embs32 = img2text(imgs32)
    fakes32 = text2img(embs32)

fig, axes = plt.subplots(4, 16, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

for i in range(32):
    row, col = divmod(i, 16)

    # Real (top 2 rows)
    if row < 2:
        ax = axes[row][col]
        img = imgs32[i].cpu().permute(1,2,0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_facecolor("#0d1117")
    # Fake (bottom 2 rows)
    ax2 = axes[row+2][col]
    img2 = fakes32[i].cpu().permute(1,2,0).numpy()
    img2 = (img2 * 0.5 + 0.5).clip(0, 1)
    ax2.imshow(img2)
    ax2.axis("off")
    ax2.set_facecolor("#0d1117")

axes[0][0].set_title("Real →", color="#8b949e", fontsize=8, loc="left")
axes[2][0].set_title("Generated →", color="#f78166", fontsize=8, loc="left")
fig.suptitle("Real vs Generated Faces — Epoch 50 (top: real, bottom: generated)",
             color="white", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("results/real_vs_generated.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved results/real_vs_generated.png")
print("All plots saved to results/")

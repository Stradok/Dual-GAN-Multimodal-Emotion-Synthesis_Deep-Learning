import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from data.fer_loader import FERDataset
from models.generators.img2text_gan import Img2TextGenerator
from models.discriminators.img2text_disc import Img2TextDiscriminator
from models.clip.clip_embedder import CLIPEncoder
from trainers.utils import save_checkpoint
from trainers.losses import GANLoss

def train_img2text(dataset_root, epochs=50, batch_size=8, device="cuda", cfg=None, checkpoint_dir="experiments/exp1/checkpoints"):
    """
    Train Image -> Text GAN
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training Image -> Text GAN on {device}")

    # Dataset with multiple workers for faster data loading
    # num_workers=0 on Windows to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    dataset = FERDataset(dataset_root, split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(num_workers > 0))

    # Models
    emb_dim = cfg["latent_dim"] if cfg else 512
    G = Img2TextGenerator(emb_dim=emb_dim).to(device)
    D = Img2TextDiscriminator(emb_dim=emb_dim).to(device)

    # CLIP encoder for ground truth text embeddings
    clip_encoder = CLIPEncoder(device)
    clip_encoder.model.eval()  # Freeze CLIP during training

    # Loss function
    criterion = GANLoss()

    # Optimizers
    opt_G = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    save_interval = cfg.get("save_interval", 10) if cfg else 10
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    G.train()
    D.train()
    
    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for batch_idx, (imgs, captions) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            
            # Get ground truth text embeddings from CLIP
            with torch.no_grad():
                real_text_embeddings = clip_encoder.encode_text(captions)
            
            # ========== Train Discriminator ==========
            opt_D.zero_grad()
            
            # Real text embeddings
            real_preds = D(real_text_embeddings)
            d_loss_real = criterion(real_preds, target_is_real=True)
            
            # Fake text embeddings (generated from images)
            with torch.no_grad():
                fake_text_embeddings = G(imgs)
            fake_preds = D(fake_text_embeddings.detach())
            d_loss_fake = criterion(fake_preds, target_is_real=False)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_D.step()
            
            # ========== Train Generator ==========
            opt_G.zero_grad()
            
            fake_text_embeddings = G(imgs)
            fake_preds = D(fake_text_embeddings)
            g_loss = criterion(fake_preds, target_is_real=True)
            
            g_loss.backward()
            opt_G.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(dataloader)}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f"[Img2Text] Epoch {epoch+1}/{epochs} done - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
        
        # Save checkpoints periodically and at the end
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            save_checkpoint(G, opt_G, epoch + 1, os.path.join(checkpoint_dir, f"img2text_generator_epoch_{epoch+1}.pth"))
            save_checkpoint(D, opt_D, epoch + 1, os.path.join(checkpoint_dir, f"img2text_discriminator_epoch_{epoch+1}.pth"))
            print(f"  Saved checkpoints to {checkpoint_dir}")

import torch
import os
import sys
import argparse
import yaml
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.fer_loader import FERDataset
from models.generators.img2text_gan import Img2TextGenerator
from models.generators.text2img_gan import Text2ImageGenerator
from models.clip.clip_embedder import CLIPEncoder
from models.clip.clip_utils import compute_clip_similarity
from torch.utils.data import DataLoader
from trainers.utils import load_model_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models using CLIP similarity")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/exp1/checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch number to load (default: latest)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = cfg["dataset_root"]
    emb_dim = cfg["img2text"]["latent_dim"]
    ngf = cfg["text2img"].get("ngf", 64)
    
    # Find checkpoint files
    checkpoint_dir = args.checkpoint_dir
    if args.epoch is None:
        # Find latest epoch
        img2text_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("img2text_generator_epoch_") and f.endswith(".pth")]
        if not img2text_files:
            print(f"Error: No checkpoints found in {checkpoint_dir}")
            return
        epochs = [int(f.split("_")[-1].split(".")[0]) for f in img2text_files]
        epoch = max(epochs)
        print(f"Loading latest checkpoint from epoch {epoch}")
    else:
        epoch = args.epoch
    
    img2text_path = os.path.join(checkpoint_dir, f"img2text_generator_epoch_{epoch}.pth")
    text2img_path = os.path.join(checkpoint_dir, f"text2img_generator_epoch_{epoch}.pth")
    
    if not os.path.exists(img2text_path) or not os.path.exists(text2img_path):
        print(f"Error: Checkpoints not found for epoch {epoch}")
        return
    
    # Load models
    print("Loading models...")
    img2text = Img2TextGenerator(emb_dim=emb_dim).to(device)
    text2img = Text2ImageGenerator(emb_dim=emb_dim, ngf=ngf).to(device)
    
    load_model_checkpoint(img2text, img2text_path, device)
    load_model_checkpoint(text2img, text2img_path, device)
    
    img2text.eval()
    text2img.eval()
    print("Models loaded successfully!")
    
    # Load CLIP model for evaluation
    print("Loading CLIP model...")
    clip_model = CLIPEncoder(device)
    
    # Load dataset
    dataset = FERDataset(dataset_root, split="test")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print("Evaluating...")
    all_similarities = []
    
    for i, (imgs, captions) in enumerate(loader):
        if args.num_samples and i * loader.batch_size >= args.num_samples:
            break
            
        imgs = imgs.to(device)
        captions_emb = img2text(imgs)
        fake_imgs = text2img(captions_emb)
        
        # Compute similarity between generated images and text captions
        sim_scores = compute_clip_similarity(fake_imgs, captions, clip_model)
        batch_similarities = sim_scores.diag().cpu().numpy()  # Diagonal = matching pairs
        all_similarities.extend(batch_similarities)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1)*loader.batch_size} samples...")
    
    # Calculate statistics
    all_similarities = np.array(all_similarities)
    mean_sim = np.mean(all_similarities)
    std_sim = np.std(all_similarities)
    min_sim = np.min(all_similarities)
    max_sim = np.max(all_similarities)
    
    print("\n" + "="*50)
    print(f"Evaluation Results (Epoch {epoch}):")
    print(f"  Mean CLIP Similarity: {mean_sim:.4f}")
    print(f"  Std Deviation: {std_sim:.4f}")
    print(f"  Min Similarity: {min_sim:.4f}")
    print(f"  Max Similarity: {max_sim:.4f}")
    print(f"  Total Samples: {len(all_similarities)}")
    print("="*50)

if __name__ == "__main__":
    main()

import torch
import os
import sys
import argparse
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.fer_loader import FERDataset
from models.generators.img2text_gan import Img2TextGenerator
from models.generators.text2img_gan import Text2ImageGenerator
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from trainers.utils import load_model_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained models")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/exp1/checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch number to load (default: latest)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="samples",
                        help="Directory to save generated images")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate")
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
        print(f"Looking for: {img2text_path}")
        print(f"Looking for: {text2img_path}")
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
    
    # Prepare dataset and output directory
    dataset = FERDataset(dataset_root, split="test")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        for i, (imgs, captions) in enumerate(loader):
            if i >= args.num_samples:
                break
            imgs = imgs.to(device)
            captions_emb = img2text(imgs)
            fake_imgs = text2img(captions_emb)
            
            # Save images
            for j in range(imgs.size(0)):
                save_image(fake_imgs[j], os.path.join(args.output_dir, f"fake_{i*4+j}_epoch{epoch}.png"), normalize=True)
                print(f"  Saved sample {i*4+j}: caption='{captions[j]}' -> fake_{i*4+j}_epoch{epoch}.png")
    
    print(f"Inference complete! Images saved to {args.output_dir}")

if __name__ == "__main__":
    main()

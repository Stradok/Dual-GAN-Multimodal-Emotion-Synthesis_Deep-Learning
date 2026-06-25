import yaml
from trainers.train_img2text import train_img2text
from trainers.train_text2img import train_text2img

if __name__ == '__main__':
    # Load config
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset_root = cfg["dataset_root"]
    device = cfg["device"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    checkpoint_dir = cfg.get("checkpoint_dir", "experiments/exp1/checkpoints")

    # Train Image->Text GAN
    train_img2text(dataset_root, epochs=epochs, batch_size=batch_size, device=device, cfg=cfg["img2text"], checkpoint_dir=checkpoint_dir)

    # Train Text->Image GAN
    train_text2img(dataset_root, epochs=epochs, batch_size=batch_size, device=device, cfg=cfg["text2img"], checkpoint_dir=checkpoint_dir)

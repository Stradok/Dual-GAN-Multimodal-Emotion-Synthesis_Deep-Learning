import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EMOTION_LABELS = {
    "angry": "a person with an angry expression",
    "disgust": "a person with a disgusted expression",
    "fear": "a person with a fearful expression",
    "happy": "a person with a happy expression",
    "neutral": "a person with a neutral expression",
    "sad": "a person with a sad expression",
    "surprise": "a person with a surprised expression",
}

class FERDataset(Dataset):
    def __init__(self, root, split="train", img_size=64):
        self.root = os.path.join(root, split)
        self.img_size = img_size
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        for emotion in sorted(os.listdir(self.root)):
            emotion_dir = os.path.join(self.root, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            caption = EMOTION_LABELS.get(emotion.lower(), f"a person with a {emotion} expression")
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(emotion_dir, fname), caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), caption

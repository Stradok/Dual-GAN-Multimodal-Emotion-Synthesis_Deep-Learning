import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder:
    def __init__(self, device, model_name="openai/clip-vit-base-patch32"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_images(self, images):
        import torch.nn.functional as F
        images = (images + 1) / 2  # [-1,1] -> [0,1]
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self.model.vision_model(pixel_values=images)
            return self.model.visual_projection(out.pooler_output)

    def encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            return self.model.text_projection(out.pooler_output)

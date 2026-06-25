import torch
from .clip_embedder import CLIPEncoder

def compute_clip_similarity(images, texts, clip_model: CLIPEncoder):
    """
    Returns similarity scores between images and texts.
    images: tensor [-1,1]
    texts: list of str
    """
    image_features = clip_model.encode_images(images)
    text_features = clip_model.encode_text(texts)
    
    # Normalize
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    similarity = torch.matmul(image_features, text_features.T)
    return similarity

from transformers import AutoImageProcessor, AutoModel
from torch.nn.functional import adaptive_avg_pool1d
from torch import no_grad

def get_encoder(encoder_id, device="cuda"):

    if "byol" in encoder_id.lower():
        print("Not implemented yet -- BYOL")
        return None, None
    
    if "moco" in encoder_id.lower():
        print("Not implemented yet -- MoCo")
        return None, None

    if "simclr" in encoder_id.lower():
        print("Not implemented yet -- MoCo")
        return None, None        
    
    image_processor = AutoImageProcessor.from_pretrained(encoder_id, use_fast=True)
    encoder = AutoModel.from_pretrained(encoder_id).to(device)

    return encoder, image_processor


def _pool_features(features, target_size):
    current_size = features.shape[-1]
    if current_size < target_size:
        msg = f"Model output size {current_size} is less than target size {target_size}"
        raise Exception(msg)
    features = features.unsqueeze(1)
    features = adaptive_avg_pool1d(features, target_size)
    features = features.squeeze(1)
    return features

def get_features(encoder, X, features_size, device="cuda"):
    X = X.to(device)
    batch_size = X.shape[0]
    is_clip = type(encoder).__name__ == "CLIPModel"
    with no_grad():
        if is_clip:
          features = encoder.get_image_features(X)
        else:
          outputs = encoder(X)
          features = outputs.pooler_output
          features = features.view(batch_size, -1)
    features = _pool_features(features, features_size)
    return features

if __name__=="main":
    # Test Only
    pass

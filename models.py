from transformers import AutoImageProcessor, AutoModel, ViTImageProcessor
from torch.nn.functional import adaptive_avg_pool1d
from torch import no_grad
import torch
import json
import timm

def get_encoder(encoder_id, device="cuda"):

    if "custom" in encoder_id.lower():

        if "byol" in encoder_id.lower():
            print("Not implemented yet -- BYOL")
        
        if "moco" in encoder_id.lower():
            checkpoint_url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
            state_dict = checkpoint["state_dict"]
            # Keep only base encoder(without head) and remove everything else (momentum, predictor)
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    state_dict[k.replace("module.base_encoder.", "")] = state_dict[k]
                del state_dict[k]
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            # Drop head from ViT model
            model.head= torch.nn.Identity()
            try:
                model.load_state_dict(state_dict)
            except:
                print("Timm model or checkpoint architecture changed")
            model.to(device)
            encoder = model
            image_processor = ViTImageProcessor()

        if "simclr" in encoder_id.lower():
            print("Not implemented yet -- MoCo")

    else:    
        image_processor = AutoImageProcessor.from_pretrained(encoder_id, use_fast=True)
        encoder = AutoModel.from_pretrained(encoder_id).to(device)

    return encoder, image_processor

def pool_features(features, to_dimensionality):
    batch_size = features.shape[0]
    current_dim = features.shape[1]

    if current_dim < to_dimensionality:
        raise Exception(f"Error: Model output dim {current_dim} is less than target dim {to_dimensionality}")
    
    if current_dim == to_dimensionality:
        return features
    
    pooled_features = adaptive_avg_pool1d(features, to_dimensionality)
    pooled_features = pooled_features.view(batch_size, to_dimensionality)
    return pooled_features


def get_features(encoder, X, target_dim, device="cuda"):
    X = X.to(device)
    batch_size = X.shape[0]
    
    with no_grad():
        
        if "clip" in str(type(encoder)):
          outputs = encoder.vision_model(X)
          features = outputs.last_hidden_layer[:, 0, :]
          features = pool_features(features, target_dim)

        elif "timm" in str(type(encoder)):
            outputs = encoder(X)
            features = pool_features(outputs, target_dim)
        
        elif "resnet" in str(type(encoder)):
            outputs = encoder(X)
            features = outputs.pooler_output
            features = features.squeeze()
            features = pool_features(features, target_dim)

        else:
            outputs = encoder(X)
            features = outputs.last_hidden_layer
            features = features[:, 0, :]
            features = features.squeeze()
            features = pool_features(features, target_dim)

    return features

def _test_encoder(encoder_id):
    batch_size = 32
    X = torch.rand((batch_size, 3, 224, 224)).to("cuda")
    encoder, img_processor = get_encoder(encoder_id)
    X = img_processor(X, return_tensors="pt")["pixel_values"]
    target_dim = 768
    features = get_features(encoder, X, target_dim=target_dim)
    assert features.shape == (batch_size, target_dim)
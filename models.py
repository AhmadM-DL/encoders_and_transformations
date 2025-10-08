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
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder.head'):
                    del state_dict[k]
                if k.startswith('module.base_encoder'):
                    state_dict[k.replace("module.base_encoder.", "")] = state_dict[k]
                    del state_dict[k]
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            msg = model.load_state_dict(state_dict)
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}
            model.to(device)
            encoder = model
            image_processor = ViTImageProcessor()

        if "simclr" in encoder_id.lower():
            print("Not implemented yet -- MoCo")

    else:    
        image_processor = AutoImageProcessor.from_pretrained(encoder_id, use_fast=True)
        encoder = AutoModel.from_pretrained(encoder_id).to(device)

    return encoder, image_processor


def get_features(encoder, X, device="cuda"):
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
    return features

def _test_encoder(encoder_id):
    batch_size = 32
    X = torch.rand((batch_size, 3, 224, 224)).to("cuda")
    encoder, img_processor = get_encoder(encoder_id)
    X = img_processor(X, return_tensors="pt")["pixel_values"]
    features = get_features(encoder, X)
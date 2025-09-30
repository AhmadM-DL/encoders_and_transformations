from transformers import AutoImageProcessor, AutoModel

def get_model(model_id, device="cuda"):

    if "byol" in model_id:
        print("Not implemented yet -- BYOL")
        return None, None
    
    if "moco" in model_id.lower():
        print("Not implemented yet -- MoCo")
        return None, None

    if "simclr" in model_id.lower():
        print("Not implemented yet -- MoCo")
        return None, None        
    
    image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)

    return model, image_processor

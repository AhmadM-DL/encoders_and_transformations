import json
from encoders import get_encoder, get_features
from datasets import get_dataset

if __name__=="main":
    config = json.load(open("./config.json", "r"))
    datasets_objs = [d for d in config["datasets"] if d["active"]]
    encoders_objs = [e for e in config["encoders"] if e["active"]]
    transformations_objs = [t for t in config["transformations"] if t["active"]]
    device = "cuda"
    for encoder_obj in encoders_objs:
        encoder_id = encoder_obj["id"]
        encoder, img_processor = get_encoder(encoder_id, device)
        for dataset_obj in datasets_objs:
            dataset_name = dataset_obj["id"]
            dataset_task = datasets_objs["classification"]
            train_dataset = get_dataset(dataset_name, dataset_task, "train", img_processor)
            test_dataset = get_dataset(dataset_name, dataset_task, "test", img_processor)
            val_dataset = get_dataset(dataset_name, dataset_task, "val", img_processor)
        

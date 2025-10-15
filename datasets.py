from torch.utils.data import Dataset
import torch
from torchvision.datasets import FGVCAircraft


class ClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, processor):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.data = self.__download_dataset__()

    def __download_dataset__(self):
        if self.dataset_name == "aircraft":
            dataset = FGVCAircraft(root='data/FGVCAircraft', split=self.split, download=True)
        else:
            raise Exception(f"Dataset {self.dataset_name} is not supported!")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.dataset_name == "aircraft":
            image, label = item[0], item[1]
        else:
            raise Exception(f"Dataset {self.dataset_name} is not supported!")
        image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        return image, label

def _mock_processor(images, return_tensors):
    images = torch.tensor(images)
    return {"pixel_values": images}
    
def _test_dataset(dataset_name):
    dataset = ClassificationDataset(dataset_name=dataset_name, split='train', processor=_mock_processor)
    print(f"Dataset: {dataset_name}, Number of samples: {len(dataset)}")
    for i in range(3):
        image, label = dataset[i]
        print(f"Sample {i}: Image shape: {image.shape}, Label: {label}")
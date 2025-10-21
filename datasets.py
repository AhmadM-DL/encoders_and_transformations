from torchvision.datasets import ImageFolder, FGVCAircraft, Flowers102
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from medmnist import INFO
from utils import download_using_axel
import zipfile
import medmnist
import torch
import os


class CUB2011Dataset(Dataset):
    """Custom CUB-200-2011 dataset"""
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    
    def __init__(self, root, split='train', download=True):
        self.root = root
        self.split = split
        
        if download and not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            download_and_extract_archive(self.url, root, extract_root=root)
        
        images_path = os.path.join(root, 'CUB_200_2011', 'images.txt')
        labels_path = os.path.join(root, 'CUB_200_2011', 'image_class_labels.txt')
        split_path = os.path.join(root, 'CUB_200_2011', 'train_test_split.txt')
        
        images_df = pd.read_csv(images_path, sep=' ', names=['img_id', 'filepath'])
        labels_df = pd.read_csv(labels_path, sep=' ', names=['img_id', 'target'])
        split_df = pd.read_csv(split_path, sep=' ', names=['img_id', 'is_train'])
        
        data = images_df.merge(labels_df, on='img_id')
        data = data.merge(split_df, on='img_id')
        
        if split == 'train':
            self.data = data[data['is_train'] == 1].reset_index(drop=True)
        else:
            self.data = data[data['is_train'] == 0].reset_index(drop=True)
        
        self.images_dir = os.path.join(root, 'CUB_200_2011', 'images')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data.iloc[idx]['filepath'])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['target'] - 1  # Convert to 0-indexed
        return image, label

class ClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, processor):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.data = self.__download_dataset__()

    def __download_dataset__(self):
        rootpath = os.environ.get('DATASET_PATH', 'data')
        path = os.path.join(rootpath, self.dataset_name)

        if self.dataset_name == "aircraft":
            dataset = FGVCAircraft(root=path, split=self.split, download=True)

        elif self.dataset_name == "flowers102":
            dataset = Flowers102(root=path, split=self.split, download=True)

        elif self.dataset_name == "cub2011":
            dataset = CUB2011Dataset(root=path, split=self.split, download=True)

        elif self.dataset_name == "retinamnist":
            dataclass = INFO[self.dataset_name]['python_class']
            if not os.path.exists(path):  os.mkdir(path)
            dataset = getattr(medmnist, dataclass)(split=self.split, download=True, root=path, as_rgb=True, size=224)
        
        elif self.dataset_name in ["chestmnist", "tissuemnist"]:
            url = f"https://zenodo.org/records/10519652/files/{self.dataset_name}_224.npz?download=1"
            download_using_axel(url, path, f"{self.dataset_name}_224.npz", 10)
            dataclass = INFO[self.dataset_name]['python_class']
            dataset = getattr(medmnist, dataclass)(split= self.split, download=False, root=path, as_rgb=True, size=224)
        
        elif self.dataset_name == "eurosat":
            url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
            download_using_axel(url, path, "EuroSAT_RGB.zip", 10)
            zip_ref = zipfile.ZipFile(os.path.join(path, "EuroSAT_RGB.zip"), 'r')
            zip_ref.extractall(path)
            dataset = ImageFolder(os.path.join(path, "EuroSAT_RGB"))
        else:
            raise Exception(f"Dataset {self.dataset_name} is not supported!")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.dataset_name in ["aircraft", "flowers102", "cub2011"]:
            image, label = item[0], item[1]
        
        elif self.dataset_name in ["retinamnist", "tissuemnist"]:
            image, label = item[0], item[1]
            label = int(label)
        
        elif self.dataset_name == "chestmnist": # Multilabel Classification
            image, label = item[0], item[1]
        
        image = self.processor(images=image, return_tensors="pt")
        image = image['pixel_values'].squeeze()
        return image, label
    
def _mock_processor(images, return_tensors):
    images = ToTensor()(images)
    return {'pixel_values': images.unsqueeze(0)}

def _test_dataset(dataset_name):
    image = Image.new('RGB', (224, 224), color = 'red')
    dataset = ClassificationDataset(dataset_name=dataset_name, split="train", processor=_mock_processor)
    print(f"Dataset size: {len(dataset)}")
    for i in range(3):
        image, label = dataset[i]
        print(f"Image shape: {image.shape}, Label: {label}")
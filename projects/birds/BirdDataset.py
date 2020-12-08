import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import dataloader as d

class BirdDataset(Dataset):
    def __init__(self, root=None, species=-1, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            self.labels, self.images = d.load_train(species=species)
        else:
            self.labels, self.images = d.load_test(species=species)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path, label = self.images[idx]
        image = d.load_image(image_path)

        return (image, label)
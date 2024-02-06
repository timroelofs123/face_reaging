import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from pathlib import Path


# Define the transformations
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomCrop(512),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, idx):
        folder_name = self.image_folders[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        # # Get the list of image filenames in the folder
        # image_filenames = [f"{i}.jpg" for i in range(0, 101, 10)]
        image_filenames = os.listdir(folder_path)

        # Pick two random assets from the folder
        source_image_name, target_image_name = random.sample(image_filenames, 2)
        # source_image_name, target_image_name = '20.jpg', '80.jpg'

        source_age = int(Path(source_image_name).stem) / 100
        target_age = int(Path(target_image_name).stem) / 100

        # Randomly select two assets from the folder
        source_image_path = os.path.join(folder_path, source_image_name)
        target_image_path = os.path.join(folder_path, target_image_name)

        source_image = Image.open(source_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')

        # Apply the same random crop and augmentations to both assets
        if self.transform:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
            torch.manual_seed(seed)
            source_image = self.transform(source_image)
            torch.manual_seed(seed)
            target_image = self.transform(target_image)

        source_age_channel = torch.full_like(source_image[:1, :, :], source_age)
        target_age_channel = torch.full_like(source_image[:1, :, :], target_age)

        # Concatenate the age channels with the source_image
        source_image = torch.cat([source_image, source_age_channel, target_age_channel], dim=0)

        return source_image, target_image

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define Fruit Dataset Class
class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.fruit_types = []
        self.colors = []
        
        # Assume subdirectories are fruit types
        for fruit_color_type in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit_color_type)
            if os.path.isdir(fruit_path):
                for img_name in os.listdir(fruit_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(fruit_path, img_name)
                        # Add image path
                        self.images.append(img_path)
                        # Extract fruit type
                        fruit_color_type_splitted = fruit_color_type.split()
                        fruit_type = fruit_color_type_splitted[0]
                        # Extract fruit color
                        fruit_color = fruit_color_type_splitted[1]
                        # Append fruit color and type to list
                        self.fruit_types.append(fruit_type)
                        self.colors.append(fruit_color)
                
        # Create dictionaries to map labels to indices
        self.fruit_type_to_idx = {fruit: idx for idx, fruit in enumerate(set(self.fruit_types))}
        self.color_to_idx = {color: idx for idx, color in enumerate(set(self.colors))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        fruit_type = self.fruit_types[idx]
        color = self.colors[idx]

        if self.transform:
            image = self.transform(image)

        return (image, 
                self.fruit_type_to_idx[fruit_type], 
                self.color_to_idx[color])
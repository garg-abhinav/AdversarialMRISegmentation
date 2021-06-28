import numpy as np
import os
import random
import itertools
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image


class im_dataset(Dataset):
    def __init__(self, images, im_size=224):
        self.images = images
        self.im_size = im_size

        self.transform = transforms.Compose([
                       transforms.Resize((im_size, im_size)),
                       transforms.ToTensor()])
    
    def __getitem__(self, idx):
        image = self.images[idx]
        save_image(image, 'img_temp.png')
        image = Image.open('img_temp.png')
        image_t = self.transform(image)
        return image_t

    def __len__(self):
        return len(self.images)


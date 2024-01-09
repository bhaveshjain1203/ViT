import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # get list of all files in image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # TODO Why convert to np.array - using albumentations lib,which when using pIL needs to be conerted to np.array
        image = np.array(Image.open(img_path).convert("RGB")) # load image,convert to RGB color mode and convert to np array
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0 # to make mask binary,set all pixel values having 255 to 1,where 1 is the region of interest

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

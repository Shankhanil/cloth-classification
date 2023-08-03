from glob import glob
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
from config import CLASSES

idx_to_class = {i:j for i, j in enumerate(CLASSES)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

class ClassificationDataset(Dataset):
    def __init__(self, image_folder, shuffle = False, transform=False):
        self.image_paths = []
        for path in glob(os.path.join(image_folder, '*/*')):
            
            self.image_paths.append(path)

        self.transform = transform

        # print(self.image_paths)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)  #["image"]
        
        return image, label

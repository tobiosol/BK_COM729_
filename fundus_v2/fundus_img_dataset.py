
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import proj_util
from fundus_v2 import fundus_img_augmentor,fundus_img_preprocessorV23
from skimage.feature import graycomatrix, graycoprops

class FundusImageDataset(Dataset):
    
    def __init__(self, image_dir, csv_file, transform=None, train=False, test=False, num_augmentations=50):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.labels_df = pd.read_csv(csv_file, index_col=0)
        self.transform = transform
        self.train = train
        self.test = test
        self.num_augmentations = num_augmentations
        self.image_paths = proj_util.load_images_from_folder(self.image_dir)
        print("image_paths", len(self.image_paths))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_preprocessor = fundus_img_preprocessorV23.FundusImagePreprocessorV23()
        self.image_augmentor = fundus_img_augmentor.FundusImageAugmentor()

    def __len__(self):
        return len(self.image_paths) * (self.num_augmentations + 1)

    def __getitem__(self, idx):
        
        if isinstance(idx, list):
            idx = idx[0]  # Handle list of indices by taking the first element
        
        original_idx = idx // (self.num_augmentations + 1)
        augmentation_idx = idx % (self.num_augmentations + 1)

        image_path = self.image_paths[original_idx]
        image = Image.open(image_path)
        label = self.get_labels(image_path)

        if self.test:
            image = self.image_preprocessor.preprocess(image=image)

        # if self.train:
        #     if augmentation_idx > 0:
        #         image = self.image_augmentor.augment_image(image, augmentation_idx - 1)

        if self.transform:
            image = self.transform(image)

        # Convert image to tensor
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).squeeze()

        
        # image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)  # Add the singleton channel dimension
        image_tensor = image_tensor.repeat(1, 1, 1)  # Repeat along the batch dimension

        # print("image_tensor", image_tensor.shape)
        
        return image_tensor, label
        
    def get_labels(self, image_filename):
        index = os.path.splitext(os.path.basename(image_filename))[0]
        self.labels_df.index = self.labels_df.index.astype(str)
        row = self.labels_df.loc[index].values.astype(np.int64)
        label_index = np.argmax(row)
        return torch.tensor(label_index, dtype=torch.long).to(self.device)
    
    
    def adjust_input_channels(self, input_tensor):
    # Check if the input tensor has more than 1 channel
        if input_tensor.shape[1] > 1:
            # Average the channels to convert to a single channel
            input_tensor = input_tensor.mean(dim=1, keepdim=True)
        return input_tensor
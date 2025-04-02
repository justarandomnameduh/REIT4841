from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
import torch

DATASET_PATH = "../CUB_200_2011/"
IAMGE_PATH = os.path.join(DATASET_PATH, "images")
NUM_CLASSES = len(os.listdir(IAMGE_PATH))  # 200 classes

# Check if dataset path exists
if not os.path.isdir(DATASET_PATH) or not os.path.isdir(IAMGE_PATH):
    raise FileNotFoundError(f"Dataset path {DATASET_PATH} or image path {IAMGE_PATH} does not exist.")

class CUBDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, display_samples=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_dir = os.path.join(root_dir, "images")

        # Load image
        images_df = pd.read_csv(os.path.join(root_dir, "images.txt"), sep=" ", header=None, names=["id", "path"])
        labels_df = pd.read_csv(os.path.join(root_dir, "image_class_labels.txt"), sep=" ", names=["id", "label"])
        # Load train/test split
        split_df = pd.read_csv(os.path.join(root_dir, "train_test_split.txt"), sep=" ", names=["id", "is_train"])

        # Merge dataframes
        self.data = images_df.merge(labels_df, on="id").merge(split_df, on="id")

        # Filter for train or test data
        if self.train:
            self.data = self.data[self.data["is_train"] == 1]
        else:
            self.data = self.data[self.data["is_train"] == 0]

        self.data = self.data.reset_index(drop=True)
        print(f"Loaded {len(self.data)} {'train' if self.train else 'test'} samples.")
        # Print first 5 rows of the dataset
        if display_samples:
            print("Sample data:")
            print(self.data.head())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.data.iloc[idx]["path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file {img_path} not found.")
            if idx + 1 < len(self.data):
                print(f"Trying next image at index {idx + 1}.")
                return self.__getitem__(idx + 1)
            else:
                raise FileNotFoundError(f"Critical: Could not load image {img_path} and no fallback.")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if idx + 1 < len(self.data):
                print(f"Trying next image at index {idx + 1}.")
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"Critical: Could not load image {img_path} and no fallback.")
            
        label = int(self.data.iloc[idx]["label"]) - 1  # Convert to 0-based index
        if self.transform:
            image = self.transform(image)
        return image, label
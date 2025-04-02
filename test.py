import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DATASET_PATH = 'CUB_200_2011' # <<<--- IMPORTANT: UPDATE THIS PATH!
IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
NUM_CLASSES = 200
BATCH_SIZE = 32       # Adjust based on your GPU memory
NUM_EPOCHS = 25       # Start with a smaller number (e.g., 10-25) for initial runs
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_LR_STEP_SIZE = 7
STEP_LR_GAMMA = 0.1   # Learning rate scheduler decay
FEATURE_EXTRACT = False # Set to True to only train the last layer, False to fine-tune more layers

# Check if the dataset path exists
if not os.path.isdir(DATASET_PATH) or not os.path.isdir(IMAGE_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATASET_PATH}'. "
        f"Please download CUB-200-2011 and update the DATASET_PATH variable."
    )

# --- Custom Dataset Class ---
class CUBDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_dir = os.path.join(self.root_dir, 'images')

        # Load image paths and IDs
        images_df = pd.read_csv(os.path.join(self.root_dir, 'images.txt'),
                                sep=' ', names=['img_id', 'filepath'])
        # Load image labels (adjusting for 0-based indexing)
        labels_df = pd.read_csv(os.path.join(self.root_dir, 'image_class_labels.txt'),
                                sep=' ', names=['img_id', 'label'])
        labels_df['label'] = labels_df['label'] - 1 # Convert 1-based to 0-based index

        # Load train/test split
        split_df = pd.read_csv(os.path.join(self.root_dir, 'train_test_split.txt'),
                               sep=' ', names=['img_id', 'is_training_img'])

        # Merge dataframes
        self.data = images_df.merge(labels_df, on='img_id')
        self.data = self.data.merge(split_df, on='img_id')

        # Filter for train or test set
        if self.train:
            self.data = self.data[self.data['is_training_img'] == 1]
        else:
            self.data = self.data[self.data['is_training_img'] == 0]

        # Reset index
        self.data = self.data.reset_index(drop=True)

        print(f"Loaded {'train' if train else 'test'} dataset with {len(self.data)} images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.data.loc[idx, 'filepath'])
        try:
            image = Image.open(img_path).convert('RGB') # Ensure image is RGB
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            # Return a dummy image and label or raise an error
            # For simplicity, let's try the next image if possible, or raise error
            # This part might need more robust handling depending on requirements
            if idx + 1 < len(self.data):
                return self.__getitem__(idx + 1)
            else:
                raise FileNotFoundError(f"Critical: Could not load image {img_path} and no fallback.")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if idx + 1 < len(self.data):
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"Critical: Could not load image {img_path} and no fallback.")


        label = int(self.data.loc[idx, 'label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Data Augmentation and Normalization ---
# Use ImageNet stats as we are using a model pre-trained on ImageNet
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # ResNet input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# --- Create Datasets and Dataloaders ---
print("Initializing Datasets and Dataloaders...")
image_datasets = {
    'train': CUBDataset(root_dir=DATASET_PATH, train=True, transform=data_transforms['train']),
    'test': CUBDataset(root_dir=DATASET_PATH, train=False, transform=data_transforms['test'])
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=4)
    for x in ['train', 'test']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
# class_names = image_datasets['train'].classes # Not directly available in custom dataset

# --- Device Configuration ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Pre-trained Model (ResNet) ---
print("Loading pre-trained ResNet model...")
# Using resnet50 as an example. You can try others like resnet18, resnet34, resnet101
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# --- Modify Model for CUB (Transfer Learning / Fine-tuning) ---
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # Freeze all layers initially

# Freeze parameters if feature extracting
set_parameter_requires_grad(model_ft, FEATURE_EXTRACT)

# Get the number of input features for the last layer
num_ftrs = model_ft.fc.in_features

# Replace the last fully connected layer with a new one for NUM_CLASSES
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Send the model to the designated device (GPU or CPU)
model_ft = model_ft.to(device)

print("Model architecture modified for CUB-200.")
# print(model_ft) # Uncomment to see the model structure

# --- Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized if feature_extract=True
# Otherwise, all parameters are optimized.
params_to_update = model_ft.parameters()
print("Params to learn:")
if FEATURE_EXTRACT:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name) # Print all trainable parameters if fine-tuning

optimizer_ft = optim.SGD(params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

# Decay LR by a factor of GAMMA every STEP_SIZE epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)

# --- Training Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            num_batches = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Print progress within epoch
                if (i + 1) % 100 == 0:
                     print(f'  [{phase}] Batch {i+1}/{num_batches} Loss: {loss.item():.4f}')


            if phase == 'train':
                scheduler.step() # Step the learning rate scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item()) # Use .item() for scalar Tensor
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model weights
                torch.save(model.state_dict(), 'cub_resnet50_best.pth')
                print(f"Saved best model weights with accuracy: {best_acc:.4f}")


        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# --- Start Training ---
print("Starting Training...")
model_trained, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                     num_epochs=NUM_EPOCHS)

# --- Save the final trained model ---
torch.save(model_trained.state_dict(), 'cub_resnet50_final.pth')
print("Final model saved to cub_resnet50_final.pth")

# --- Plot Training History ---
def plot_history(history):
    epochs = range(len(history['train_loss']))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Convert accuracy tensors to numbers if they aren't already
    train_acc = [acc for acc in history['train_acc']]
    val_acc = [acc for acc in history['val_acc']]
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")
    # plt.show() # Uncomment to display plot immediately

plot_history(history)

print("Script finished.")
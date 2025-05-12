import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
from tqdm import tqdm

try:
    from dataloader import CUBDataset
except ImportError:
    print("Error: Could not import CUBDataset. Ensure the dataloader.py file is in the same directory.")
    exit()

try:
    from model import VGG16_200
except ImportError:
    print("Error: Could not import VGG16_200. Ensure the model.py file is in the same directory.")
    exit()

DATASET_PATH = "../CUB_200_2011/"
NUM_CLASSES = 200
INPUT_SIZE = 224  # Image size for VGG16
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
OPTIMIZER_TYPE = "Adam"

# Scheduler parameters
USE_SCHEDULER = True
SCHEDULER_TYPE = "StepLR"
STEP_LR_STEP_SIZE = 15
STEP_LR_GAMMA = 0.1

NUM_WORKERS = 4
MODEL_SAVE_NAME = 'cub_vgg16_200_best.pth'
PLOT_SAVE_NAME = 'cub_vgg16_200_loss.png'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("--- Configuration ---")
print(f"Dataset Path: {DATASET_PATH}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Num Epochs: {NUM_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Optimizer: {OPTIMIZER_TYPE}")
print(f"Using Scheduler: {USE_SCHEDULER} ({SCHEDULER_TYPE})")
print("--------------------")

# ImageNet stats
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Create datasets and dataloaders
try:
    image_datasets = {
        'train': CUBDataset(root_dir=DATASET_PATH, train=True, transform=data_transforms['train'], display_samples=True),
        'test': CUBDataset(root_dir=DATASET_PATH, train=False, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print("Datasets and Dataloaders created successfully.")
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Test dataset size: {dataset_sizes['test']}")
except Exception as e:
    print(f"Error creating datasets or dataloaders: {e}")
    exit()

# Initialize the model

model = VGG16_200(num_classes=NUM_CLASSES, init_weights=True).to(device)
pretrained_model = models.vgg16(weights='DEFAULT')
for param in pretrained_model.parameters():
    param.requires_grad = False  # Freeze all layers

num_ftrs = pretrained_model.classifier[6].in_features
pretrained_model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
pretrained_model = pretrained_model.to(device)
print("Model initialized successfully.")

criterion = nn.CrossEntropyLoss()
params_to_update = model.parameters()

if OPTIMIZER_TYPE == "Adam":
    optimizer = torch.optim.Adam(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.Adam(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = None
if USE_SCHEDULER:
    if SCHEDULER_TYPE == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)

# Train function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        # TQDM description for the epoch
        epoch_desc = f"Epoch {epoch + 1}/{num_epochs}" # Use epoch+1 for 1-based display
        print(epoch_desc)
        print('-' * len(epoch_desc)) # Match separator to description length

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0 # Keep track of total samples processed in the phase

            progress_bar = tqdm(dataloaders[phase],
                                desc=f"  [{phase.capitalize():<5}]", # Left align phase name
                                leave=False,
                                ncols=100) # Adjust ncols for desired bar width

            # Iterate over data using the tqdm wrapper
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0) # Get current batch size
                total_samples += batch_size

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
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

                # Update tqdm postfix with running loss and accuracy
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

            # Calculate final epoch statistics after the loop completes
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Print final epoch summary (after progress bar finishes)
            print(f'  {phase.capitalize()} Final Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                 # Step the StepLR scheduler after the training phase (if used)
                if USE_SCHEDULER and SCHEDULER_TYPE.lower() == 'steplr':
                    scheduler.step()
                    # Optional: print LR update
                    print(f"    LR stepped to: {scheduler.get_last_lr()}")

            else: # 'test' phase acts as validation
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Step the ReduceLROnPlateau scheduler based on validation loss
                if USE_SCHEDULER and SCHEDULER_TYPE.lower() == 'reducelronplateau':
                     scheduler.step(epoch_loss) # Pass val_loss

                # Deep copy the model if it's the best validation accuracy so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model weights
                    torch.save(model.state_dict(), MODEL_SAVE_NAME)
                    print(f"    => Saved new best model weights to {MODEL_SAVE_NAME} (Acc: {best_acc:.4f})")


        print() # Newline after each epoch completes both phases

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, save_path):
    epochs = range(len(history['train_loss']))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    train_acc = [acc for acc in history['train_acc']]
    val_acc = [acc for acc in history['val_acc']]
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

if __name__ == "__main__":
    # Train the model
    model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # Plot training history
    plot_history(history, "scratch_" + PLOT_SAVE_NAME)

    model, history = train_model(pretrained_model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # Plot training history
    plot_history(history, "pretrained_" + PLOT_SAVE_NAME)

    print("Training completed and model saved.")
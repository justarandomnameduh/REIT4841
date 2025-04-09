import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import os
import numpy as np
import pickle
from PIL import Image
import time
import argparse
import matplotlib.pyplot as plt

from model import ConceptBottleneckModel

def parse_args():
    parser = argparse.ArgumentParser(description='CBM Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--num_classes', type=int, default=200, help='Number of classes')
    parser.add_argument('--num_concepts', type=int, default=312, help='Number of concepts')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--concept_loss_weight', type=float, default=0.01, help='Weight multiplier for concept loss')
    parser.add_argument('--no_pretrained', action='store_true', help='Do not pretrain the model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the model and logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    return parser.parse_args()

class CUBPickleDataset(Dataset):
    """
    Load data from the .pkl file for CUB-200
    """
    def __init__(self, pkl_file_path, transform=None, concept_key='uncertain_attr_label'):
        try:
            with open(pkl_file_path, 'rb') as f:
                self.metadata_list = pickle.load(f)
            print(f"Successfully loaded {len(self.metadata_list)} samples from {pkl_file_path}")
        except Exception as e:
            print(f"Error loading {pkl_file_path}: {e}")
            self.metadata_list = []

        self.transform = transform
        self.concept_key = concept_key
        self.num_concepts = len(self.metadata_list[0][self.concept_key]) if self.metadata_list else 0
        
    def __len__(self):
        return len(self.metadata_list)
    
    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        image_path = metadata['path']
        class_label = metadata['label']
        concepts = torch.tensor(metadata[self.concept_key], dtype=torch.float32)
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found at {image_path}. Returning placeholder.")
            placeholder_size = (224, 224)
            image = Image.new('RGB', placeholder_size, (0, 0, 0))
            class_label = -1 # bad sample indicator
            concepts = torch.zeros(self.num_concepts, dtype=torch.float32)
            
        if self.transform:
            image = self.transform(image)
        
        return image, class_label, concepts
    

def train_epoch(model, loader, optimizer, criterion_class, criterion_concept, device, concept_loss_weight):
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_concept_loss = 0.0

    correct_class = 0
    total_samples = 0
    correct_concepts_threshold = 0
    total_concepts = 0

    start_time = time.time()
    for i, (inputs, labels, concepts) in enumerate(loader):
        valid_indices = labels != -1
        if not valid_indices.all():
            original_count = inputs.size(0)
            inputs = inputs[valid_indices]
            labels = labels[valid_indices]
            concepts = concepts[valid_indices]
            if inputs.size(0) == 0:
                continue # All images failed
            print(f"Warning: Skipped {original_count - inputs.size(0)} samples in batch {i} due to loading errors.")

        inputs, labels, concepts = inputs.to(device), labels.to(device), concepts.to(device)

        optimizer.zero_grad()
        class_logits, concept_logits = model(inputs)

        loss_class = criterion_class(class_logits, labels)
        loss_concept = criterion_concept(concept_logits, concepts)

        total_loss = loss_class + concept_loss_weight * loss_concept
        total_loss.backward()
        optimizer.step()

        # Display stats
        batch_size = inputs.size(0)
        # .item() to only use raw data for logging, do not need gradient
        running_loss += total_loss.item() * batch_size
        running_class_loss += loss_class.item() * batch_size
        running_concept_loss += loss_concept.item() * batch_size

        _, predicted_class = torch.max(class_logits.data, 1)
        total_samples += batch_size
        correct_class += (predicted_class == labels).sum().item()

        # Concept accuracy (thresholding)
        predicted_concepts_binary = (torch.sigmoid(concept_logits) > 0.5).float()
        ground_truth_concepts_binary = (concepts > 0.5).float()

        # Calculate concept accuracy
        correct_concepts_threshold += (predicted_concepts_binary == ground_truth_concepts_binary).sum().item()
        total_concepts += concepts.numel()

        if (i + 1) % 50 == 0:
            print(f"Batch {(i+1)}/{len(loader)} | Loss: {total_loss.item():.4f}")

    
    epoch_duration = time.time() - start_time
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_class_loss = running_class_loss / total_samples if total_samples > 0 else 0
    epoch_concept_loss = running_concept_loss / total_samples if total_samples > 0 else 0
    epoch_class_acc = correct_class / total_samples if total_samples > 0 else 0
    epoch_concept_acc = correct_concepts_threshold / total_concepts if total_concepts > 0 else 0

    print(f"Epoch train time: {epoch_duration:.2f}s")
    return epoch_loss, epoch_class_loss, epoch_concept_loss, epoch_class_acc, epoch_concept_acc



def evaluate(model, loader, criterion_class, criterion_concept, device, concept_loss_weight):
    model.eval()
    running_loss = 0.0
    running_class_loss = 0.0
    running_concept_loss = 0.0

    correct_class = 0
    total_samples = 0
    correct_concepts_threshold = 0
    total_concepts = 0

    with torch.no_grad():
        for inputs, labels, concepts in loader:
            valid_indices = labels != -1
            if not valid_indices.all():
                inputs = inputs[valid_indices]
                labels = labels[valid_indices]
                concepts = concepts[valid_indices]
                if inputs.size(0) == 0:
                    continue

            inputs, labels, concepts = inputs.to(device), labels.to(device), concepts.to(device)

            class_logits, concept_logits = model(inputs)

            loss_class = criterion_class(class_logits, labels)
            loss_concept = criterion_concept(concept_logits, concepts)
            total_loss = loss_class + concept_loss_weight * loss_concept

            batch_size = inputs.size(0)
            running_loss += total_loss.item() * batch_size
            running_class_loss += loss_class.item() * batch_size
            running_concept_loss += loss_concept.item() * batch_size

            _, predicted_class = torch.max(class_logits.data, 1)
            total_samples += batch_size
            correct_class += (predicted_class == labels).sum().item()

            # Concept accuracy (thresholding)
            predicted_concepts_binary = (torch.sigmoid(concept_logits) > 0.5).float()
            ground_truth_concepts_binary = (concepts > 0.5).float()

            correct_concepts_threshold += (predicted_concepts_binary == ground_truth_concepts_binary).sum().item()
            total_concepts += concepts.numel()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_class_loss = running_class_loss / total_samples if total_samples > 0 else 0
    epoch_concept_loss = running_concept_loss / total_samples if total_samples > 0 else 0
    epoch_class_acc = correct_class / total_samples if total_samples > 0 else 0
    epoch_concept_acc = correct_concepts_threshold / total_concepts if total_concepts > 0 else 0
    return epoch_loss, epoch_class_loss, epoch_concept_loss, epoch_class_acc, epoch_concept_acc


def plot_metrics(history, output_dir):
    """
    Plot training and validation metrics
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-darkgrid')

    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 6))
    # Plot total loss
    ax_loss.plot(epochs, history['train_loss'], 'bo-', label='Train Total Loss')
    ax_loss.plot(epochs, history['val_loss'], 'ro-', label='Validation total Loss')
    
    ax_loss.plot(epochs, history['train_class_loss'], 'b--', label='Train Class Loss')
    ax_loss.plot(epochs, history['val_class_loss'], 'r--', label='Val Class Loss')
    ax_loss.plot(epochs, history['train_concept_loss'], 'b:', label='Train Concept Loss')
    ax_loss.plot(epochs, history['val_concept_loss'], 'r:', label='Val Concept Loss')

    ax_loss.set_title('Training and Validation Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    loss_plot_path = os.path.join(output_dir, 'loss_curves.png')
    fig_loss.savefig(loss_plot_path)
    print(f"Loss curves saved to {loss_plot_path}")
    plt.close(fig_loss)



    ### Plot accuracies
    fig_acc, ax_acc = plt.subplots(1, 1, figsize=(10, 6)) # Single plot for accuracies

    # Plot Class Accuracy
    ax_acc.plot(epochs, history['train_class_acc'], 'go-', label='Train Class Accuracy')
    ax_acc.plot(epochs, history['val_class_acc'], 'mo-', label='Val Class Accuracy')

    # Plot Concept Accuracy
    ax_acc.plot(epochs, history['train_concept_acc'], 'g--', label='Train Concept Accuracy (Thresh)')
    ax_acc.plot(epochs, history['val_concept_acc'], 'm--', label='Val Concept Accuracy (Thresh)')

    ax_acc.set_title('Training and Validation Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim(bottom=max(0, ax_acc.get_ylim()[0]-0.05), top=min(1.0, ax_acc.get_ylim()[1]+0.05)) # Adjust y-axis limits
    ax_acc.legend()
    ax_acc.grid(True)

    acc_plot_path = os.path.join(output_dir, 'accuracy_curves.png')
    fig_acc.savefig(acc_plot_path)
    print(f"Accuracy curves saved to {acc_plot_path}")
    plt.close(fig_acc)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Define transformation - use ImageNet mean and std
    INPUT_SIZE = 224
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor_transform = transforms.ToTensor()
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            to_tensor_transform,
            normalize_transform
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            to_tensor_transform,
            normalize_transform
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            to_tensor_transform,
            normalize_transform
        ])
    }

    # Load data
    try:
        
        train_dataset = CUBPickleDataset(pkl_file_path=os.path.join(args.data_dir, 'train_data.pkl'), transform=data_transforms['train'])
        val_dataset = CUBPickleDataset(os.path.join(args.data_dir, 'val_data.pkl'), transform=data_transforms['val'])
        test_dataset = CUBPickleDataset(os.path.join(args.data_dir, 'test_data.pkl'), transform=data_transforms['test'])
        print("Datasets created.")

        if args.num_concepts != train_dataset.num_concepts and train_dataset.num_concepts > 0:
            print(f"Warning: --num-concepts argument ({args.num_concepts}) does not match" 
                  f"number of concepts found in dataset ({train_dataset.num_concepts}). "
                  f"Using value from dataset: {train_dataset.num_concepts}")
            args.num_concepts = train_dataset.num_concepts

        print("Create dataloaders...")
        pin_memory = True if DEVICE == torch.device("cuda") else False
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        print("Dataloaders created.")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")

        # for samples, target in train_loader:
        #     print(f"Sample : {samples} | Target: {target}")
        #     break

    except Exception as e:
        print(f"\n An error occurred during data setup: {e}.\nExiting.")
        exit()

    # Initialize model
    model = ConceptBottleneckModel(num_classes=args.num_classes, num_concepts=args.num_concepts, pretrained=(not args.no_pretrained))
    model.to(DEVICE)
    print("Model initialized.")

    criterion_class = nn.CrossEntropyLoss()
    criterion_concept = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    training_history = {
        'train_loss': [], 'train_class_loss': [], 'train_concept_loss': [],
        'train_class_acc': [], 'train_concept_acc': [],
        'val_loss': [], 'val_class_loss': [], 'val_concept_loss': [],
        'val_class_acc': [], 'val_concept_acc': []
    }
    best_val_class_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        train_loss, train_class_loss, train_concept_loss, train_class_acc, train_concept_acc = train_epoch(
            model, train_loader, optimizer, criterion_class, criterion_concept, DEVICE, args.concept_loss_weight
        )
        print(f"Train Loss: {train_loss:.4f} | Class Loss: {train_class_loss:.4f} | Concept Loss: {train_concept_loss:.4f}")
        print(f"Train Class Accuracy: {train_class_acc:.4f} | Train Concept Accuracy: {train_concept_acc:.4f}")
        
        val_loss, val_class_loss, val_concept_loss, val_class_acc, val_concept_acc = evaluate(
            model, val_loader, criterion_class, criterion_concept, DEVICE, args.concept_loss_weight
        )
        print(f"Validation Loss: {val_loss:.4f} | Class Loss: {val_class_loss:.4f} | Concept Loss: {val_concept_loss:.4f}")
        print(f"Validation Class Accuracy: {val_class_acc:.4f} | Validation Concept Accuracy: {val_concept_acc:.4f}")

        training_history['train_loss'].append(train_loss)
        training_history['train_class_loss'].append(train_class_loss)
        training_history['train_concept_loss'].append(train_concept_loss)
        training_history['train_class_acc'].append(train_class_acc)
        training_history['train_concept_acc'].append(train_concept_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_class_loss'].append(val_class_loss)
        training_history['val_concept_loss'].append(val_concept_loss)
        training_history['val_class_acc'].append(val_class_acc)
        training_history['val_concept_acc'].append(val_concept_acc)

        if val_class_acc > best_val_class_acc:
            best_val_class_acc = val_class_acc
            model_save_path = os.path.join(args.output_dir, f'cbm_joint_{args.concept_loss_weight}_best_model.pth')
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model checkpoint to {model_save_path} (Val Acc: {best_val_class_acc*100:.2f}%)")
        print("-" * 25)
    
    print(f"Best validation class accuracy: {best_val_class_acc:.4f}.\nTraining complete.")

    print("\nGenerating training plots...")
    plot_metrics(training_history, args.output_dir)

    # Sanity check - load the best model and evaluate on test set
    # COMMENT THIS SECTION WHEN BULK TRAINING
    best_model_path = os.path.join(args.output_dir, f'cbm_joint_{args.concept_loss_weight}_best_model.pth')
    if os.path.exists(best_model_path):
        final_model = ConceptBottleneckModel(num_classes=args.num_classes, num_concepts=args.num_concepts, pretrained=False)
        print(f"Loaded best model from {best_model_path}.")
        final_model.load_state_dict(torch.load(best_model_path))
        final_model.to(DEVICE)
        final_model.eval()
        test_loss, test_class_loss, test_concept_loss, test_class_acc, test_concept_acc = evaluate(
            final_model, test_loader, criterion_class, criterion_concept, DEVICE, args.concept_loss_weight
        )
        print(f"Test Loss: {test_loss:.4f} | Class Loss: {test_class_loss:.4f} | Concept Loss: {test_concept_loss:.4f}")
        print(f"Test Class Accuracy: {test_class_acc:.4f} | Test Concept Accuracy: {test_concept_acc:.4f}")
    else:
        print(f"Best model checkpoint not found at {best_model_path}. Skipping final evaluation.")
        
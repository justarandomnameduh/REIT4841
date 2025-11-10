import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dnn.model import EfficientNetV2
from ham10000_dataset import HAM10000Dataset
from derm7pt_dataset import Derm7ptDataset
from cub_dataset import CUBDataset


class MultiResolutionTransform:
    """
    Data augmentation with rotation and flips.
    Uses fixed 224x224 resolution for batching compatibility.
    """
    
    def __init__(self, mode='train'):
        self.mode = mode
        
        if mode == 'train':
            self.base_transforms = [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        else:
            self.base_transforms = []
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, img):
        resolution = 224
        
        transform = transforms.Compose(
            self.base_transforms + [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                self.normalize
            ]
        )
        
        return transform(img)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and exponential decay.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, decay_rate=0.97, decay_every=2.4):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            epochs_since_warmup = self.current_epoch - self.warmup_epochs
            lr = self.base_lr * (self.decay_rate ** (epochs_since_warmup / self.decay_every))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def train_epoch(model, dataloader, criterion, optimizer, device, task_type='class'):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        task_type: 'class' or 'concept'
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        if task_type == 'class':
            images, labels = batch
        else:
            images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if task_type == 'class':
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        else:
            loss = criterion(outputs, labels)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    f1 = None
    if task_type == 'class':
        f1 = f1_score(all_targets, all_preds, average='weighted')
    else:
        f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }


def validate_epoch(model, dataloader, criterion, device, task_type='class'):
    """
    Validate for one epoch.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        task_type: 'class' or 'concept'
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            if task_type == 'class':
                images, labels = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if task_type == 'class':
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
            else:
                loss = criterion(outputs, labels)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.numel()
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_targets.extend(labels.cpu().numpy().flatten())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    f1 = None
    if task_type == 'class':
        f1 = f1_score(all_targets, all_preds, average='weighted')
    else:
        f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }


def plot_training_results(history, save_path, task_type='class'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        task_type: 'class' or 'concept'
    """
    sns.set_style('whitegrid')
    
    if task_type == 'class':
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score (Weighted)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy (Note: Imbalanced)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score (Better Metric)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(args):
    """
    Main training function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.dataset == 'ham10000':
        train_dataset = HAM10000Dataset(
            root=args.data_root,
            split='train',
            mode=args.task,
            transform=MultiResolutionTransform(mode='train')
        )
        val_dataset = HAM10000Dataset(
            root=args.data_root,
            split='val',
            mode=args.task,
            transform=MultiResolutionTransform(mode='val')
        )
        num_outputs = train_dataset.num_classes if args.task == 'class' else train_dataset.num_concepts
    elif args.dataset == 'derm7pt':
        train_dataset = Derm7ptDataset(
            root=args.data_root,
            split='train',
            mode=args.task,
            transform=MultiResolutionTransform(mode='train')
        )
        val_dataset = Derm7ptDataset(
            root=args.data_root,
            split='valid',
            mode=args.task,
            transform=MultiResolutionTransform(mode='val')
        )
        num_outputs = train_dataset.num_classes if args.task == 'class' else train_dataset.num_concepts
    elif args.dataset == 'cub':
        train_dataset = CUBDataset(
            root=args.data_root,
            split='train',
            mode=args.task,
            transform=MultiResolutionTransform(mode='train')
        )
        val_dataset = CUBDataset(
            root=args.data_root,
            split='test',
            mode=args.task,
            transform=MultiResolutionTransform(mode='val')
        )
        num_outputs = train_dataset.num_classes if args.task == 'class' else train_dataset.num_concepts
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f'Dataset: {args.dataset}')
    print(f'Task: {args.task}')
    print(f'Number of outputs: {num_outputs}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    model = EfficientNetV2(
        num_outputs=num_outputs, 
        dropout=args.dropout,
        pretrained=args.pretrained
    )
    
    if args.pretrained_path is not None:
        print(f'Loading pre-trained weights from {args.pretrained_path}')
        model.load_pretrained_backbone(args.pretrained_path, num_outputs)
        print('Pre-trained weights loaded successfully. Final layer replaced.')
    
    model.to(device)
    
    if args.task == 'class':
        # Apply class weights for imbalanced datasets
        if args.dataset == 'derm7pt':
            # Derm7pt: 73.99% benign, 26.01% malignant (ratio ~2.84:1)
            # Weight the minority class more heavily
            class_weights = torch.tensor([1.0, 2.84]).to(device)
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=args.label_smoothing
            )
        else:
            # HAM10000, CUB, or other datasets - use label smoothing only
            criterion = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing
            )
    else:
        # Concept prediction task - use weighted BCE loss
        if args.dataset == 'ham10000':
            pos_weight = torch.tensor([4.2] * num_outputs).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif args.dataset == 'derm7pt':
            pos_weight = torch.tensor([3.0] * num_outputs).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif args.dataset == 'cub':
            pos_weight = torch.tensor([3.4] * num_outputs).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        alpha=0.9,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        decay_rate=0.97,
        decay_every=2.4
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    best_val_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        lr = scheduler.step()
        history['lr'].append(lr)
        
        print(f'\nEpoch {epoch + 1}/{args.epochs} (LR: {lr:.6f})')
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.task)
        val_metrics = validate_epoch(model, val_loader, criterion, device, args.task)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        if args.task == 'class':
            history['val_f1'].append(val_metrics['f1'])
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, Val F1: {val_metrics['f1']:.4f}")
            
            val_metric = val_metrics['f1']
        else:
            history['val_f1'].append(val_metrics['f1'])
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, Train F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, Val F1: {val_metrics['f1']:.4f}")
            
            val_metric = val_metrics['f1']
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f'Saved best model with metric: {best_val_metric:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{args.patience}')
            
            if patience_counter >= args.patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    print(f'\nTraining completed!')
    print(f'Best epoch: {best_epoch}')
    print(f'Best validation metric: {best_val_metric:.4f}')
    
    plot_training_results(history, args.plot_path, args.task)
    print(f'Saved training plot to {args.plot_path}')


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNetV2 on skin lesion datasets')
    
    parser.add_argument('--dataset', type=str, required=True, choices=['ham10000', 'derm7pt', 'cub'],
                        help='Dataset to use')
    parser.add_argument('--task', type=str, required=True, choices=['class', 'concept'],
                        help='Task type: class or concept prediction')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the best model')
    parser.add_argument('--plot_path', type=str, required=True,
                        help='Path to save the training plot')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pre-trained model weights for transfer learning')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pre-trained weights (default: True)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch without ImageNet pre-training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor for classification (default: 0.1)')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=350,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Base learning rate (default: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
    
    train_model(args)


if __name__ == '__main__':
    main()

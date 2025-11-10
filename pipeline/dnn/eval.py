import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dnn.model import EfficientNetV2
from ham10000_dataset import HAM10000Dataset
from derm7pt_dataset import Derm7ptDataset
from cub_dataset import CUBDataset


def get_evaluation_transform():
    """
    Standard transform for evaluation.
    """
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings from a dataloader.
    
    Args:
        model: The trained model
        dataloader: Dataloader to extract embeddings from
        device: Device to use
    
    Returns:
        List of embeddings (numpy arrays)
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            if len(batch) == 2:
                images, _ = batch
            else:
                images, _, _ = batch
            
            images = images.to(device)
            batch_embeddings = model.get_embedding(images)
            embeddings.extend(batch_embeddings.cpu().numpy())
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained EfficientNetV2')
    
    parser.add_argument('--dataset', type=str, required=True, choices=['ham10000', 'derm7pt', 'cub'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--task', type=str, required=True, choices=['class', 'concept'],
                        help='Task type used for training')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for extraction')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform = get_evaluation_transform()
    
    if args.dataset == 'ham10000':
        splits = ['train', 'val', 'test']
    elif args.dataset == 'derm7pt':
        splits = ['train', 'valid', 'test']
    else:
        splits = ['train', 'test']
    
    for split in splits:
        print(f'\nProcessing {split} split...')
        
        if args.dataset == 'ham10000':
            dataset = HAM10000Dataset(
                root=args.data_root,
                split=split,
                mode='both',
                transform=transform
            )
            num_outputs = dataset.num_classes if args.task == 'class' else dataset.num_concepts
        elif args.dataset == 'derm7pt':
            dataset = Derm7ptDataset(
                root=args.data_root,
                split=split,
                mode='both',
                transform=transform
            )
            num_outputs = dataset.num_classes if args.task == 'class' else dataset.num_concepts
        else:
            dataset = CUBDataset(
                root=args.data_root,
                split=split,
                mode='both',
                transform=transform
            )
            num_outputs = dataset.num_classes if args.task == 'class' else dataset.num_concepts
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        if split == splits[0]:
            model = EfficientNetV2(num_outputs=num_outputs, dropout=0.2)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            print(f'Loaded model from {args.model_path}')
        
        embeddings = extract_embeddings(model, dataloader, device)
        
        if args.dataset == 'derm7pt' and split == 'valid':
            split_name = 'val'
        else:
            split_name = split
        
        output_path = os.path.join(args.output_dir, f'{split_name}.pkl')
        
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f'Saved {len(embeddings)} embeddings to {output_path}')


if __name__ == '__main__':
    main()

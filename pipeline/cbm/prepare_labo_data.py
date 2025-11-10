"""
Prepare LaBo-compatible dataset from concept extractor outputs.

This script converts concept files from concept_extractor to LaBo format.
Fixes the issue where c=1 but num_concept=48 by ensuring proper concept counts.
"""

import json
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# HAM10000 class names (binary classification: mel vs nv)
HAM10000_CLASSES = sorted(['mel', 'nv'])

def load_concept_file(concept_file: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load concept file and extract concepts.
    
    Returns:
        all_concepts: List of all unique concepts
        class_concepts: Dict mapping class names to their concepts
    """
    with open(concept_file, 'r') as f:
        data = json.load(f)
    
    # Get all unique concepts and class-specific concepts
    all_concepts = data.get('all', [])
    class_concepts = data.get('class_concepts', {})
    
    return all_concepts, class_concepts


def create_labo_class2concepts(class_concepts: Dict[str, List[str]], 
                                 classes: List[str]) -> Dict[str, List[str]]:
    """
    Create LaBo-compatible class2concepts dictionary.
    
    Ensures all classes are present and properly mapped.
    """
    labo_class2concepts = {}
    
    for class_name in classes:
        if class_name in class_concepts:
            labo_class2concepts[class_name] = class_concepts[class_name]
        else:
            # If class has no concepts, use empty list
            labo_class2concepts[class_name] = []
            print(f"Warning: Class '{class_name}' has no concepts!")
    
    return labo_class2concepts


def create_labo_npy_files(class2concepts: Dict[str, List[str]], 
                           save_dir: Path) -> Dict[str, int]:
    """
    Create LaBo-required .npy files: concepts_raw.npy, cls_names.npy, concept2cls.npy
    
    Returns:
        stats: Dictionary with concept count statistics
    """
    # Sort class names to ensure consistent ordering
    class_names = sorted(list(class2concepts.keys()))
    
    # Build concept list and concept-to-class mapping
    all_concepts = []
    concept2cls = []
    
    for class_idx, class_name in enumerate(class_names):
        concepts = class2concepts[class_name]
        for concept in concepts:
            all_concepts.append(concept)
            concept2cls.append(class_idx)
    
    # Convert to numpy arrays
    concepts_raw = np.array(all_concepts, dtype=object)
    cls_names = np.array(class_names, dtype=object)
    concept2cls_arr = np.array(concept2cls, dtype=np.int32)
    
    # Save files
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / 'concepts_raw.npy', concepts_raw)
    np.save(save_dir / 'cls_names.npy', cls_names)
    np.save(save_dir / 'concept2cls.npy', concept2cls_arr)
    
    # Calculate statistics
    total_concepts = len(all_concepts)
    concepts_per_class = {cls: len(class2concepts[cls]) for cls in class_names}
    
    stats = {
        'total_concepts': total_concepts,
        'num_classes': len(class_names),
        'concepts_per_class': concepts_per_class,
        'avg_concepts_per_class': total_concepts / len(class_names) if class_names else 0
    }
    
    return stats


def create_split_files(dataset_root: Path, output_dir: Path, dataset: str = 'ham10000'):
    """
    Create LaBo-compatible split files: class2images_{train,val,test}.p
    
    These are pickle files mapping class names to lists of image paths.
    Creates balanced 70/15/15 train/val/test splits for binary classification (mel vs nv).
    
    Note: Image paths are stored relative to the LaBo datasets folder, not absolute paths.
    LaBo will look for images at: datasets/{dataset_name}/images/{image_id}.jpg
    But we'll use symlinks to point to the actual dataset location.
    """
    if dataset == 'ham10000':
        # Original HAM10000 dataset location
        original_images_root = Path('/home/nqmtien/REIT4841/datasets/ham10000/images')
        
        # Use existing groundtruth CSVs (mel vs nv binary classification)
        embeddings_root = Path('/home/nqmtien/REIT4841/pipeline/dnn/embeddings/ham10000')
        
        # Load all splits to get complete image-to-class mapping
        image_to_class = {}
        
        for split in ['train', 'val', 'test']:
            pkl_file = embeddings_root / f'{split}.pkl'
            if not pkl_file.exists():
                print(f"Warning: {pkl_file} not found")
                continue
            
            # The embeddings pkl files are lists of embeddings, but we need metadata
            # Let's check the ham10000_dataset.py for the structure
            pass
        
        # Load the HAM10000Dataset class to load binary classification data (mel vs nv)
        import sys
        sys.path.insert(0, '/home/nqmtien/REIT4841/pipeline_old')
        from ham10000_dataset import HAM10000Dataset
        
        # Get image-class mappings from the dataset class
        for split_name in ['train', 'val', 'test']:
            try:
                dataset_obj = HAM10000Dataset(
                    root='/home/nqmtien/REIT4841/datasets/ham10000',
                    split=split_name,
                    mode='class'
                )
                
                for idx in range(len(dataset_obj)):
                    row = dataset_obj.df.iloc[idx]
                    image_id = row['image_id']
                    dx = row['dx']
                    image_to_class[image_id] = dx
                
            except Exception as e:
                print(f"Warning: Could not load {split_name} split: {e}")
                continue
        
        if not image_to_class:
            print("Error: Could not load any image-to-class mappings")
            # Create empty splits
            for split_name in ['train', 'val', 'test']:
                class2images = {cls: [] for cls in HAM10000_CLASSES}
                output_path = output_dir / f'class2images_{split_name}.p'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    pickle.dump(class2images, f)
            return
        
        print(f"Loaded {len(image_to_class)} image-to-class mappings")
        
        # Group images by class
        class_to_images = {cls: [] for cls in HAM10000_CLASSES}
        for image_id, class_name in image_to_class.items():
            if class_name in class_to_images:
                # Store just the filename: {image_id}.jpg
                # LaBo will append this to img_path which already ends with /images
                rel_path = f'{image_id}.jpg'
                class_to_images[class_name].append(rel_path)
        
        # Create splits for each class (70% train, 15% val, 15% test)
        np.random.seed(42)
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for class_name, images in class_to_images.items():
            images = np.array(images)
            n_images = len(images)
            
            if n_images == 0:
                print(f"Warning: No images for class {class_name}")
                splits['train'][class_name] = []
                splits['val'][class_name] = []
                splits['test'][class_name] = []
                continue
            
            # Shuffle
            indices = np.random.permutation(n_images)
            images = images[indices]
            
            # Split
            n_train = int(0.7 * n_images)
            n_val = int(0.15 * n_images)
            
            splits['train'][class_name] = images[:n_train].tolist()
            splits['val'][class_name] = images[n_train:n_train+n_val].tolist()
            splits['test'][class_name] = images[n_train+n_val:].tolist()
        
        # Save split files
        for split_name in ['train', 'val', 'test']:
            class2images = splits[split_name]
            
            output_path = output_dir / f'class2images_{split_name}.p'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(class2images, f)
            
            total_images = sum(len(v) for v in class2images.values())
            print(f"Created {split_name} split: {total_images} images")
            
            # Show per-class breakdown
            for cls in sorted(class2images.keys()):
                count = len(class2images[cls])
                print(f"  {cls:10s}: {count:4d} images")


def prepare_single_config(k: int, n: int, c: int, vlm: str, dataset: str = 'ham10000'):
    """
    Prepare LaBo dataset for a single hyperparameter configuration.
    """
    # Paths
    concept_extractor_root = Path('/home/nqmtien/REIT4841/pipeline/concept_extractor')
    cbm_root = Path('/home/nqmtien/REIT4841/pipeline/cbm')
    
    concept_file = concept_extractor_root / f'concepts/{vlm}/{dataset}/concept_all_{k}_{n}_{c}.json'
    output_dir = cbm_root / f'datasets/{dataset}_k{k}_n{n}_c{c}_{vlm}'
    
    # Check if concept file exists
    if not concept_file.exists():
        print(f"Error: Concept file not found: {concept_file}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Preparing: k={k}, n={n}, c={c}, vlm={vlm}")
    print(f"{'='*80}")
    
    # Load concepts
    all_concepts, class_concepts = load_concept_file(concept_file)
    
    print(f"Loaded concepts:")
    print(f"  - Total unique concepts: {len(all_concepts)}")
    print(f"  - Classes with concepts: {len(class_concepts)}")
    
    # Create LaBo format
    if dataset == 'ham10000':
        classes = HAM10000_CLASSES
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    labo_class2concepts = create_labo_class2concepts(class_concepts, classes)
    
    # Save class2concepts.json
    concepts_dir = output_dir / 'concepts'
    concepts_dir.mkdir(parents=True, exist_ok=True)
    
    with open(concepts_dir / 'class2concepts.json', 'w') as f:
        json.dump(labo_class2concepts, f, indent=2)
    
    print(f"\nSaved: {concepts_dir / 'class2concepts.json'}")
    
    # Create .npy files
    stats = create_labo_npy_files(labo_class2concepts, concepts_dir)
    
    print(f"\nConcept Statistics:")
    print(f"  - Total concepts: {stats['total_concepts']}")
    print(f"  - Number of classes: {stats['num_classes']}")
    print(f"  - Avg concepts per class: {stats['avg_concepts_per_class']:.2f}")
    print(f"\nPer-class breakdown:")
    for cls, count in stats['concepts_per_class'].items():
        print(f"    {cls:10s}: {count:3d} concepts")
    
    min_expected = c
    max_expected = len(classes) * c
    if min_expected <= stats['total_concepts'] <= max_expected:
        print(f"\nConcept count in valid range: {stats['total_concepts']} concepts")
        print(f"   (expected range: {min_expected} to {max_expected} for c={c})")
    else:
        print(f"\n  WARNING: Unusual concept count: {stats['total_concepts']}")
        print(f"   Expected range: {min_expected}-{max_expected} for c={c}")
    
    classes_without_concepts = [cls for cls, count in stats['concepts_per_class'].items() if count == 0]
    if classes_without_concepts:
        print(f"\n  WARNING: Classes without concepts: {classes_without_concepts}")
    else:
        print(f"All {len(classes)} classes have concepts")
    
    # Create split files
    splits_dir = output_dir / 'splits'
    create_split_files(Path('/home/nqmtien/REIT4841/datasets'), splits_dir, dataset)
    
    # Create symlink to actual images directory
    images_symlink = output_dir / 'images'
    if not images_symlink.exists():
        original_images = Path(f'/home/nqmtien/REIT4841/datasets/{dataset}/images')
        if original_images.exists():
            images_symlink.symlink_to(original_images)
            print(f"\nCreated symlink: {images_symlink} -> {original_images}")
        else:
            print(f"\nWarning: Original images not found: {original_images}")
    
    print(f"\nCompleted: {output_dir}")
    print(f"{'='*80}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LaBo dataset from concept extractor outputs'
    )
    
    parser.add_argument('--dataset', default='ham10000', 
                       help='Dataset name (default: ham10000)')
    parser.add_argument('--k', type=int, nargs='+',
                       help='Cluster values (e.g., --k 1 3 5 10)')
    parser.add_argument('--n', type=int, nargs='+',
                       help='Image per cluster values (e.g., --n 1 2 4)')
    parser.add_argument('--c', type=int, nargs='+',
                       help='Concept per cluster values (e.g., --c 1 3 5 10 30)')
    parser.add_argument('--vlm', type=str, nargs='+',
                       help='VLM models (e.g., --vlm gemini qwen)')
    parser.add_argument('--all', action='store_true',
                       help='Prepare all hyperparameter combinations')
    
    args = parser.parse_args()
    
    # Define default hyperparameter ranges
    all_k = [1, 3, 5, 10]
    all_n = [1, 2, 4]
    all_c = [1, 3, 5, 10, 30, 50, 100, 200]
    all_vlm = ['gemini'] # , 'qwen'
    
    # Determine which configurations to prepare
    if args.all:
        k_values = all_k
        n_values = all_n
        c_values = all_c
        vlm_values = all_vlm
    else:
        k_values = args.k if args.k else all_k
        n_values = args.n if args.n else all_n
        c_values = args.c if args.c else all_c
        vlm_values = args.vlm if args.vlm else all_vlm
    
    # Prepare each configuration
    total = len(k_values) * len(n_values) * len(c_values) * len(vlm_values)
    success = 0
    failed = []
    
    print(f"\n{'='*80}")
    print(f"Preparing {total} configurations for {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"k values: {k_values}")
    print(f"n values: {n_values}")
    print(f"c values: {c_values}")
    print(f"vlm values: {vlm_values}")
    print(f"{'='*80}\n")
    
    for k in k_values:
        for n in n_values:
            for c in c_values:
                for vlm in vlm_values:
                    try:
                        if prepare_single_config(k, n, c, vlm, args.dataset):
                            success += 1
                        else:
                            failed.append((k, n, c, vlm))
                    except Exception as e:
                        print(f"Error preparing k={k}, n={n}, c={c}, vlm={vlm}: {e}")
                        failed.append((k, n, c, vlm))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed configurations:")
        for k, n, c, vlm in failed:
            print(f"  - k={k}, n={n}, c={c}, vlm={vlm}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from umap import UMAP
from sklearn.cluster import KMeans


def load_embeddings(dataset_name, split='train'):
    """Load embeddings and corresponding image paths from pickle files."""
    base_path = Path(__file__).parent.parent
    embedding_path = base_path / 'dnn' / 'embeddings' / dataset_name / f'{split}.pkl'
    
    embeddings = pickle.load(open(embedding_path, 'rb'))
    embeddings = np.array(embeddings)
    
    if dataset_name == 'ham10000':
        csv_path = base_path.parent / 'datasets' / dataset_name / 'groundtruth' / f'{split}.csv'
        df = pd.read_csv(csv_path)
        image_dir = base_path.parent / 'datasets' / dataset_name / 'images'
        image_paths = [str(image_dir / f"{row['image_id']}.jpg") for _, row in df.iterrows()]
    elif dataset_name == 'derm7pt':
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        actual_split = split_map.get(split, split)
        
        dataset_root = base_path.parent / 'datasets' / dataset_name / 'release_v0'
        meta_path = dataset_root / 'meta' / 'meta.csv'
        indexes_path = dataset_root / 'meta' / f'{actual_split}_indexes.csv'
        
        df_full = pd.read_csv(meta_path)
        indexes = list(pd.read_csv(indexes_path)['indexes'])
        df = df_full.iloc[indexes].reset_index(drop=True)
        
        DIAGNOSIS_MAPPING = {
            'blue nevus': 0, 'clark nevus': 0, 'combined nevus': 0, 'congenital nevus': 0,
            'dermal nevus': 0, 'recurrent nevus': 0, 'reed or spitz nevus': 0,
            'melanoma': 1, 'melanoma (in situ)': 1, 'melanoma (less than 0.76 mm)': 1,
            'melanoma (0.76 to 1.5 mm)': 1, 'melanoma (more than 1.5 mm)': 1, 'melanoma metastasis': 1,
        }
        valid_diagnoses = set(DIAGNOSIS_MAPPING.keys())
        df = df[df['diagnosis'].isin(valid_diagnoses)].reset_index(drop=True)
        
        image_dir = dataset_root / 'images'
        image_paths = [str(image_dir / row['derm']) for _, row in df.iterrows()]
    elif dataset_name == 'cub':
        dataset_root = base_path.parent / 'datasets' / 'CUB_200_2011'
        
        # Load images.txt
        images_df = pd.read_csv(
            dataset_root / 'images.txt',
            sep=' ',
            header=None,
            names=['image_id', 'image_path']
        )
        
        # Load train_test_split.txt
        split_df = pd.read_csv(
            dataset_root / 'train_test_split.txt',
            sep=' ',
            header=None,
            names=['image_id', 'is_train']
        )
        
        # Merge and filter by split
        df = images_df.merge(split_df, on='image_id')
        is_train = 1 if split == 'train' else 0
        df = df[df['is_train'] == is_train].reset_index(drop=True)
        
        # Get full image paths
        image_dir = dataset_root / 'images'
        image_paths = [str(image_dir / row['image_path']) for _, row in df.iterrows()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return embeddings, image_paths


def apply_umap_kmeans(embeddings, n_components=3, n_clusters=15, random_state=42):
    """Apply UMAP dimensionality reduction followed by K-means clustering."""
    umap_reducer = UMAP(n_components=n_components, random_state=random_state, n_jobs=1)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    return reduced_embeddings, cluster_labels, kmeans


def create_cluster_json(image_paths, cluster_labels, n_clusters):
    """Create cluster JSON structure with cluster_id and list of image paths."""
    clusters = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_images = [image_paths[i] for i in range(len(image_paths)) if cluster_mask[i]]
        
        clusters.append({
            'cluster_id': cluster_id,
            'images': cluster_images
        })
    
    return clusters


def main():
    parser = argparse.ArgumentParser(description='Apply UMAP and K-means clustering to embeddings')
    parser.add_argument('--dataset', type=str, required=True, choices=['ham10000', 'derm7pt', 'cub'],
                        help='Dataset name')
    parser.add_argument('--d', type=int, default=3, help='UMAP dimension')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    
    args = parser.parse_args()
    
    embeddings, image_paths = load_embeddings(args.dataset, args.split)
    
    reduced_embeddings, cluster_labels, kmeans = apply_umap_kmeans(
        embeddings, n_components=args.d, n_clusters=args.k
    )
    
    clusters = create_cluster_json(image_paths, cluster_labels, args.k)
    
    output_dir = Path(__file__).parent / 'clusters' / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{args.k}.json'
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)
    
    intermediate_dir = output_dir / 'intermediate'
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    intermediate_data = {
        'reduced_embeddings': reduced_embeddings.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'image_paths': image_paths
    }
    
    intermediate_file = intermediate_dir / f'{args.k}_intermediate.json'
    with open(intermediate_file, 'w') as f:
        json.dump(intermediate_data, f, indent=2)


if __name__ == '__main__':
    main()

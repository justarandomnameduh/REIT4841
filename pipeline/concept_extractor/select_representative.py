import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
np.random.seed(42)


def load_intermediate_data(dataset, k):
    """Load intermediate clustering data containing reduced embeddings and centroids."""
    base_path = Path(__file__).parent
    intermediate_file = base_path / 'clusters' / dataset / 'intermediate' / f'{k}_intermediate.json'
    
    with open(intermediate_file, 'r') as f:
        data = json.load(f)
    
    reduced_embeddings = np.array(data['reduced_embeddings'])
    cluster_labels = np.array(data['cluster_labels'])
    centroids = np.array(data['centroids'])
    image_paths = data['image_paths']
    
    return reduced_embeddings, cluster_labels, centroids, image_paths


def find_representative_images(reduced_embeddings, cluster_labels, centroids, image_paths, n_representatives=4):
    """Find n closest images to each cluster centroid, ensuring global uniqueness."""
    n_clusters = len(centroids)
    representatives = []
    used_images = set()  # Track globally used images across all clusters
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_images = [image_paths[i] for i in range(len(image_paths)) if cluster_mask[i]]
        cluster_embeddings = reduced_embeddings[cluster_mask]
        
        if len(cluster_images) == 0:
            representatives.append({
                'cluster_id': cluster_id,
                'representative_images': []
            })
            continue
        
        centroid = centroids[cluster_id:cluster_id+1]
        distances = cdist(centroid, cluster_embeddings, metric='euclidean')[0]
        
        # Sort indices by distance
        sorted_indices = np.argsort(distances)
        
        # Select n unique images that haven't been used yet
        representative_images = []
        for idx in sorted_indices:
            img_path = cluster_images[idx]
            if img_path not in used_images:
                representative_images.append(img_path)
                used_images.add(img_path)
                if len(representative_images) == n_representatives:
                    break
        
        representatives.append({
            'cluster_id': cluster_id,
            'representative_images': representative_images
        })
    
    return representatives


def main():
    parser = argparse.ArgumentParser(description='Select representative images for each cluster')
    parser.add_argument('--dataset', type=str, required=True, choices=['ham10000', 'derm7pt', 'cub'],
                        help='Dataset name')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--n', type=int, required=True, help='Number of representative images per cluster')
    
    args = parser.parse_args()
    
    reduced_embeddings, cluster_labels, centroids, image_paths = load_intermediate_data(
        args.dataset, args.k
    )
    
    representatives = find_representative_images(
        reduced_embeddings, cluster_labels, centroids, image_paths, args.n
    )
    
    output_dir = Path(__file__).parent / 'clusters' / args.dataset
    output_file = output_dir / f'representative_{args.k}_{args.n}.json'
    
    with open(output_file, 'w') as f:
        json.dump(representatives, f, indent=2)


if __name__ == '__main__':
    main()

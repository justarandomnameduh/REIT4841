import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import sys
import warnings
from matplotlib.lines import Line2D  # For custom legend

# --- Configuration ---
PROCESSED_CONCEPTS_FILE = "processed_concepts.json"
VAL_CSV_FILE = "../ham_concept_dataset/Datasets/metadata/val.csv"
OUTPUT_DIR = "."  # Directory to save any outputs
OUTPUT_TSNE_FILENAME = "senTrans-tsne_visualization.png"

# --- Helper Functions ---
def load_processed_concepts(file_path):
    """Loads processed concepts from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Filter out empty dictionary entries that might exist from previous processing steps
        data = [item for item in data if isinstance(item, dict) and item.get("image_id") and item.get("label")]
        if not data:
            print(f"Warning: No valid entries (with image_id and label) found in {file_path} after filtering.")
            return []
        print(f"Successfully loaded {len(data)} valid entries from {file_path}")
        return data
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def get_concept_embeddings(concepts_list, sentence_model):
    """
    Generate embeddings for a list of concepts using a sentence transformer model.
    
    Args:
        concepts_list (list): List of text concepts to embed
        sentence_model: Loaded SentenceTransformer model
        
    Returns:
        np.ndarray: Matrix of embeddings (n_concepts x embedding_dim)
    """
    if not concepts_list:
        print("Warning: Empty concept list provided for embedding.")
        return np.array([])
    
    print(f"Generating embeddings for {len(concepts_list)} concepts...")
    
    try:
        # Generate embeddings in a single batch
        embeddings = sentence_model.encode(concepts_list, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.array([])

def extract_key_phrases_and_labels(data):
    """
    Extract key phrases and corresponding labels from processed concepts data.
    
    Args:
        data (list): List of processed concept dictionaries
        
    Returns:
        tuple: (all_key_phrases, all_labels, all_image_ids) lists
    """
    all_key_phrases = []
    all_labels = []
    all_image_ids = []
    
    for item in data:
        if item.get("key_phrases") and item.get("label"):
            if len(item["key_phrases"]) > 0:  # Only include items with non-empty key_phrases
                all_key_phrases.append(item["key_phrases"])
                all_labels.append(item["label"])
                all_image_ids.append(item["image_id"])
    
    return all_key_phrases, all_labels, all_image_ids

def visualize_with_tsne(embeddings, labels, image_ids, output_path):
    """
    Visualize embeddings using t-SNE with labels as colors.
    
    Args:
        embeddings (np.ndarray): Matrix of concept embeddings
        labels (list): List of corresponding labels (MEL/NV)
        image_ids (list): List of corresponding image IDs
        output_path (str): Path to save the visualization
    """
    if embeddings.shape[0] == 0:
        print("Cannot visualize empty embeddings.")
        return
        
    print("Performing t-SNE dimensionality reduction...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create a color map for labels
    unique_labels = list(set(labels))
    color_map = {'MEL': 'red', 'NV': 'blue'}
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Plot each point
    for i, (x, y) in enumerate(embeddings_2d):
        label = labels[i]
        color = color_map.get(label, 'gray')
        plt.scatter(x, y, color=color, alpha=0.7)
    
    # Add labels for a sample of points to avoid overcrowding
    sample_indices = np.linspace(0, len(embeddings_2d) - 1, min(20, len(embeddings_2d)), dtype=int)
    for i in sample_indices:
        plt.annotate(
            image_ids[i],
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    # Add a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='MEL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='NV')
    ]
    plt.legend(handles=legend_elements)
    
    # Add title and labels
    plt.title('t-SNE Visualization of Concept Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"t-SNE visualization saved to {output_path}")
    plt.close()

def calculate_concept_embeddings_from_phrases(all_key_phrases, sentence_model):
    """
    Calculate embeddings from lists of key phrases by averaging.
    
    Args:
        all_key_phrases (list): List of lists, where each inner list contains key phrases
        sentence_model: SentenceTransformer model
        
    Returns:
        np.ndarray: Matrix of concept embeddings
    """
    # Flatten all phrases to get unique ones and create embeddings
    all_unique_phrases = set()
    for phrases in all_key_phrases:
        all_unique_phrases.update(phrases)
    
    unique_phrases_list = list(all_unique_phrases)
    
    if not unique_phrases_list:
        print("No unique phrases found. Cannot create embeddings.")
        return np.array([])
    
    print(f"Found {len(unique_phrases_list)} unique key phrases.")
    
    # Generate embeddings for all unique phrases
    phrase_embeddings = get_concept_embeddings(unique_phrases_list, sentence_model)
    if phrase_embeddings.shape[0] == 0:
        return np.array([])
    
    # Create a dictionary for quick lookup
    phrase_to_embedding = {phrase: embedding for phrase, embedding in zip(unique_phrases_list, phrase_embeddings)}
    
    # Calculate average embedding for each list of key phrases
    concept_embeddings = []
    for phrases in all_key_phrases:
        if phrases:
            # Get embeddings for each phrase in the list
            embeddings = [phrase_to_embedding[phrase] for phrase in phrases if phrase in phrase_to_embedding]
            if embeddings:
                # Average the embeddings
                avg_embedding = np.mean(embeddings, axis=0)
                concept_embeddings.append(avg_embedding)
            else:
                print(f"Warning: No embeddings found for phrases: {phrases}")
                # Add a zero vector as a placeholder
                concept_embeddings.append(np.zeros(phrase_embeddings.shape[1]))
        else:
            # Add a zero vector for empty phrase lists
            concept_embeddings.append(np.zeros(phrase_embeddings.shape[1]))
    
    return np.array(concept_embeddings)

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_concepts_path = os.path.join(script_dir, PROCESSED_CONCEPTS_FILE)
    output_tsne_path = os.path.join(script_dir, OUTPUT_TSNE_FILENAME)
    
    # Check if NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    # Load processed concepts data
    print(f"Loading processed concepts from {processed_concepts_path}")
    data = load_processed_concepts(processed_concepts_path)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Extract key phrases and labels
    key_phrases_lists, labels, image_ids = extract_key_phrases_and_labels(data)
    if not key_phrases_lists:
        print("No key phrases found in the data. Exiting.")
        return
    
    print(f"Extracted key phrases for {len(key_phrases_lists)} images.")
    print(f"Labels distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Load sentence transformer model
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
    except ImportError:
        print("Error: SentenceTransformer library not found. Please install it with:")
        print("pip install -U sentence-transformers")
        return
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        return
    
    # Get embeddings for concepts
    concept_embeddings = calculate_concept_embeddings_from_phrases(key_phrases_lists, model)
    if concept_embeddings.shape[0] == 0:
        print("Failed to generate concept embeddings. Exiting.")
        return
    
    print(f"Generated concept embeddings with shape {concept_embeddings.shape}")
    
    # Visualize embeddings using t-SNE
    visualize_with_tsne(concept_embeddings, labels, image_ids, output_tsne_path)

if __name__ == "__main__":
    main()

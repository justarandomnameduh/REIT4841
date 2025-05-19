import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# sentence_transformers will be imported later to allow for a graceful error message if not installed

# --- Configuration ---
PROCESSED_CONCEPTS_FILE = "processed_concepts.json"

VAL_CSV_FILE = "../ham_concept_dataset/Datasets/metadata/val.csv"

OUTPUT_DIR = "." # Directory to save any outputs if generated later
IMAGE_TRAINING_DIR = os.path.expanduser("../ham_concept_dataset/ISIC2018_Task3_Training_Input")

# --- Helper Functions ---
def load_processed_concepts(file_path):
    """Loads processed concepts from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Filter out empty dictionary entries that might exist from previous processing steps
        data = [item for item in data if isinstance(item, dict) and item.get("image_id")]
        if not data:
            print(f"Warning: No valid entries (with image_id) found in {file_path} after filtering.")
            return []
        print(f"Successfully loaded {len(data)} valid entries from {file_path}")
        return data
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def load_ground_truth(file_path):
    """
    Loads ground truth labels from a CSV file.
    Tries to intelligently find image ID and label columns.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Attempt to identify image_id column (case-insensitive)
        image_id_col_found = None
        possible_id_cols = [col for col in df.columns if 'isic' in col.lower() or 'image' in col.lower() or 'id' in col.lower()]
        if possible_id_cols:
            image_id_col_found = possible_id_cols[0] # Take the first likely candidate
            df = df.rename(columns={image_id_col_found: 'image_id'})
            print(f"Identified '{image_id_col_found}' as image ID column.")
        else:
            print(f"Error: Could not automatically identify an image_id column in {file_path}.")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Please ensure your CSV file has a clear image identifier column (e.g., 'image_id', 'ISIC_ID').")
            return None
        
        # Standardize image_id to string, as it often is in concept files
        df['image_id'] = df['image_id'].astype(str)

        # Logic for label columns
        if 'MEL' in df.columns and 'NV' in df.columns:
            print("Found 'MEL' and 'NV' columns for labels.")
            # Ensure MEL and NV are numeric, coercing errors to NaN, then fill NaN with 0
            df['MEL'] = pd.to_numeric(df['MEL'], errors='coerce').fillna(0)
            df['NV'] = pd.to_numeric(df['NV'], errors='coerce').fillna(0)
            conditions = [
                (df['MEL'] == 1),
                (df['NV'] == 1)
            ]
            choices = ['MEL', 'NV']
            df['true_label'] = np.select(conditions, choices, default='OTHER')
        elif 'label' in df.columns:
            print("Found 'label' column for ground truth.")
            df = df.rename(columns={'label': 'true_label'})
        elif 'ground_truth' in df.columns:
            print("Found 'ground_truth' column for labels.")
            df = df.rename(columns={'ground_truth': 'true_label'})
        else: # Try to find a single column that might be the label if not MEL/NV
            potential_label_cols = [col for col in df.columns if col.lower() not in ['image_id', image_id_col_found.lower()] and df[col].nunique() < 10] # Heuristic: few unique values
            if len(potential_label_cols) == 1:
                label_col_found = potential_label_cols[0]
                print(f"Warning: Did not find standard label columns. Using '{label_col_found}' as true_label column based on heuristics.")
                df = df.rename(columns={label_col_found: 'true_label'})
            else:
                print(f"Error: Could not find expected label columns ('MEL' & 'NV', or 'label', or 'ground_truth') in {file_path}.")
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Please ensure your CSV has appropriate columns for ground truth labels.")
                return None
        
        df = df[['image_id', 'true_label']]
        print(f"Successfully loaded and processed {len(df)} ground truth entries from {file_path}")
        # print(f"First 5 ground truth entries:\n{df.head()}")
        return df
    except Exception as e:
        print(f"Error loading or processing ground truth file {file_path}: {e}")
        return None

def merge_data(concepts_data, truth_df):
    """Merges concepts data with ground truth labels based on image_id."""
    if not concepts_data:
        print("No concepts data to merge.")
        return []
    if truth_df is None:
        print("Ground truth DataFrame is None. Skipping merge. True labels will be 'UNKNOWN'.")
        for entry in concepts_data:
            entry['true_label'] = 'UNKNOWN'
        return concepts_data

    # Ensure image_id in concepts_data is also string for robust merging
    for entry in concepts_data:
        if 'image_id' in entry:
            entry['image_id'] = str(entry['image_id'])

    truth_map = pd.Series(truth_df.true_label.values, index=truth_df.image_id).to_dict()

    merged_data = []
    found_matches = 0
    unmatched_concept_ids = 0
    for concept_entry in concepts_data:
        image_id = concept_entry.get("image_id")
        if image_id: # image_id should exist due to pre-filtering in load_processed_concepts
            true_label = truth_map.get(image_id)
            if true_label:
                concept_entry["true_label"] = true_label
                found_matches += 1
            else:
                concept_entry["true_label"] = "UNKNOWN"
                unmatched_concept_ids +=1
            merged_data.append(concept_entry)
        # No else needed as entries without image_id are filtered out earlier
    
    print(f"Merged data for {len(merged_data)} entries.")
    print(f"Found {found_matches} matching true labels for image_ids.")
    if unmatched_concept_ids > 0:
        print(f"Warning: {unmatched_concept_ids} image_ids from processed_concepts.json did not have a match in {VAL_CSV_FILE}.")
        if found_matches == 0 and len(merged_data) > 0:
             print("Critical Warning: No image_ids matched between the concepts file and the ground truth CSV. "
                   "Please check that the image_id formats are identical in both files (e.g., 'ISIC_0024306' vs 'ISIC_0024306.jpg').")
    return merged_data

# --- Similarity Definitions ---

# 1. Semantic Similarity (Word/Phrase Embeddings using Sentence Transformers)
def get_concept_embeddings(concepts_list, model):
    """Generates embeddings for a list of concepts using the provided SentenceTransformer model."""
    if not concepts_list:
        return np.array([])
    print(f"Generating embeddings for {len(concepts_list)} unique concepts...")
    embeddings = model.encode(concepts_list, convert_to_tensor=False, show_progress_bar=False) 
    return embeddings

def calculate_semantic_similarity_for_concept(target_concept, target_embedding, all_concepts_list, all_embeddings, top_n=5):
    """Calculates cosine similarity of a target concept to all other concepts and returns top N,
    excluding neighbors that contain the target_concept text or its word variations (case-insensitive)."""
    try:
        import nltk
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        
        # Try to ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("NLTK resources not found. Attempting to download...")
            nltk.download('punkt')
            nltk.download('wordnet')
        
        # Set up stemmer and lemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        # Get stem and lemmas of the target concept
        target_tokens = word_tokenize(target_concept.lower())
        target_stems = [stemmer.stem(word) for word in target_tokens]
        target_lemmas = [lemmatizer.lemmatize(word) for word in target_tokens]
        
    except (ImportError, LookupError) as e:
        print(f"Warning: NLTK stemming/lemmatization not available ({e}). Only exact matches will be filtered.")
        target_stems = []
        target_lemmas = []
        
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.reshape(1, -1)  # Reshape if it's a 1D array
    
    similarities = cosine_similarity(target_embedding, all_embeddings)[0]
    
    # Get indices of top N similar concepts, excluding the concept itself
    # Argsort sorts in ascending order, so we take from the end.
    sorted_indices = np.argsort(similarities)
    
    top_neighbors = []
    # Iterate from the end of sorted_indices
    for i in reversed(sorted_indices):
        # Stop if we have enough neighbors
        if len(top_neighbors) >= top_n:
            break
        
        current_concept = all_concepts_list[i]
        
        # Exclude the concept itself
        if current_concept == target_concept:
            continue

        # Exclude neighbors that contain the target_concept's text (case-insensitive)
        if target_concept.lower() in current_concept.lower():
            continue
        
        # Check for word variations using stemming/lemmatization
        try:
            if target_stems or target_lemmas:  # Only if NLTK was successfully loaded
                current_tokens = word_tokenize(current_concept.lower())
                current_stems = [stemmer.stem(word) for word in current_tokens]
                current_lemmas = [lemmatizer.lemmatize(word) for word in current_tokens]
                
                # Skip if any stem or lemma from target appears in current concept's stems/lemmas
                should_skip = False
                for stem in target_stems:
                    if any(stem == current_stem for current_stem in current_stems):
                        should_skip = True
                        break
                        
                if not should_skip:
                    for lemma in target_lemmas:
                        if any(lemma == current_lemma for current_lemma in current_lemmas):
                            should_skip = True
                            break
                            
                if should_skip:
                    continue
        except Exception as e:
            # If there's any error in the stemming/lemmatization process, 
            # just use the basic string matching approach
            pass
            
        top_neighbors.append((current_concept, similarities[i]))
        
    return top_neighbors

# 2. Co-occurrence within Descriptions
def calculate_cooccurrence(all_concepts_data, concept_field='noun_phrases'):
    """
    Calculates co-occurrence of concepts within the same descriptions.
    Args:
        all_concepts_data: List of dicts, where each dict has an image_id and a list of concepts.
        concept_field: The key in the dict that holds the list of concepts (e.g., 'noun_phrases', 'keywords').
    Returns:
        A pandas DataFrame representing the co-occurrence matrix (concept pairs as index/columns).
    """
    print(f"\n--- Calculating Co-occurrence for '{concept_field}' ---")
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    valid_entries_for_cooccurrence = 0

    for item in all_concepts_data:
        concepts = item.get(concept_field, [])
        if isinstance(concepts, list) and len(concepts) > 1:
            valid_entries_for_cooccurrence +=1
            # Ensure concepts are unique within the description for this counting method if desired,
            # or allow multiple counts if a concept appears multiple times with others.
            # For simplicity, using unique concepts per description:
            unique_concepts_in_item = sorted(list(set(c.strip() for c in concepts if isinstance(c, str) and c.strip())))
            
            for i in range(len(unique_concepts_in_item)):
                for j in range(i + 1, len(unique_concepts_in_item)):
                    # The pair is always (concept_alpha_sorted_1, concept_alpha_sorted_2)
                    c1, c2 = tuple(sorted((unique_concepts_in_item[i], unique_concepts_in_item[j])))
                    cooccurrence_counts[c1][c2] += 1
                    # If you want a symmetric matrix directly, also do:
                    # cooccurrence_counts[c2][c1] += 1 
                    # But pandas DataFrame handles this well from one-sided counts.
    
    if not cooccurrence_counts:
        print(f"No concept pairs found for co-occurrence analysis in '{concept_field}'. "
              f"Ensure entries have multiple concepts in the '{concept_field}' list.")
        return None

    print(f"Processed {valid_entries_for_cooccurrence} entries for co-occurrence calculation.")
    cooccurrence_df = pd.DataFrame(cooccurrence_counts).fillna(0)
    # For a symmetric matrix where df[A][B] == df[B][A]
    # cooccurrence_df = cooccurrence_df.add(cooccurrence_df.T, fill_value=0)
    # np.fill_diagonal(cooccurrence_df.values, 0) # Optional: zero out self-cooccurrence if it arises

    if cooccurrence_df.empty:
        print("Co-occurrence matrix is empty.")
        return None
        
    print(f"Co-occurrence matrix generated with shape: {cooccurrence_df.shape}")
    return cooccurrence_df

# 3. Contextual Similarity (Sentence Embeddings)
def calculate_contextual_similarity(all_concepts_data, sentence_model, concept_field='noun_phrases'):
    """
    Calculates contextual similarity of concepts based on the sentences they appear in.
    Each concept's contextual embedding is the average of embeddings of sentences it's found in.
    Args:
        all_concepts_data: List of dicts from merged data.
        sentence_model: Pre-loaded SentenceTransformer model.
        concept_field: Key for accessing concepts (e.g., 'noun_phrases') in each data item.
    Returns:
        A dictionary mapping unique concepts to their contextual embeddings.
        Or None if sentence_model is not available or no concepts are processed.
    """
    print("\n--- 3. Contextual Similarity (Sentence Embeddings) ---")
    if not sentence_model:
        print("Sentence model not available. Skipping contextual similarity calculation.")
        return None

    # Ensure NLTK's 'punkt' tokenizer is available for sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Attempting download...")
        try:
            nltk.download('punkt', quiet=False)
            nltk.data.find('tokenizers/punkt') # Verify
            print("Successfully downloaded 'punkt'.")
        except Exception as e:
            print(f"Failed to download 'punkt'. Error: {e}")
            print("Contextual similarity calculation will be skipped as sentence tokenization is unavailable.")
            return None

    concept_to_sentence_texts = defaultdict(list)
    processed_items = 0

    print(f"Processing items to map concepts from '{concept_field}' to sentences in 'original_reason'...")
    for item in all_concepts_data:
        original_reason = item.get('original_reason', "")
        if isinstance(original_reason, list):
            original_reason = " ".join(original_reason) # Join if reason is a list of strings
        
        if not original_reason or not isinstance(original_reason, str) or not original_reason.strip():
            continue

        concepts_in_item = item.get(concept_field, [])
        if not isinstance(concepts_in_item, list) or not concepts_in_item:
            continue

        # Deduplicate concepts within the current item for mapping
        unique_concepts_in_item = set()
        for c in concepts_in_item:
            if isinstance(c, str) and c.strip():
                 unique_concepts_in_item.add(c.strip())

        if not unique_concepts_in_item:
            continue

        try:
            sentences = sent_tokenize(original_reason)
        except Exception as e:
            print(f"Error tokenizing sentences for item {item.get('image_id', 'Unknown')}: {e}. Skipping this item for contextual similarity.")
            continue
            
        item_mapped_flag = False
        for concept_text in unique_concepts_in_item:
            for sentence in sentences:
                if concept_text.lower() in sentence.lower(): # Case-insensitive check for concept in sentence
                    concept_to_sentence_texts[concept_text].append(sentence)
                    item_mapped_flag = True
        if item_mapped_flag:
            processed_items +=1

    if not concept_to_sentence_texts:
        print(f"No concepts from '{concept_field}' were found within their 'original_reason' sentences after processing {processed_items} items.")
        return None
    
    print(f"Found {len(concept_to_sentence_texts)} unique concepts with associated sentences from {processed_items} processed items.")

    concept_contextual_embeddings = {}
    print("Generating contextual embeddings for concepts...")
    for i, (concept, sentences_containing_concept) in enumerate(concept_to_sentence_texts.items()):
        if not sentences_containing_concept: # Should not happen if logic above is correct
            continue
        # Encode all sentences where this concept appeared
        # Use set to avoid re-encoding identical sentences if a concept appears multiple times in same sentence list
        unique_sentences = list(set(sentences_containing_concept))
        sentence_embeddings = sentence_model.encode(unique_sentences, show_progress_bar=False)
        if sentence_embeddings.ndim == 2 and sentence_embeddings.shape[0] > 0:
            # The contextual embedding for the concept is the mean of its sentence embeddings
            concept_contextual_embeddings[concept] = np.mean(sentence_embeddings, axis=0)
        if (i + 1) % 100 == 0:
            print(f"  Generated contextual embeddings for {i+1}/{len(concept_to_sentence_texts)} concepts...")
    
    print(f"Finished generating contextual embeddings for {len(concept_contextual_embeddings)} concepts.")

    if not concept_contextual_embeddings:
        print("No contextual embeddings could be generated.")
        return None

    # Demonstrate similarity for a few example concepts using their contextual embeddings
    if len(concept_contextual_embeddings) >= 2:
        print("\nExample contextual similarities:")
        # Get a few concept names and their embeddings
        example_concepts = list(concept_contextual_embeddings.keys())[:min(20, len(concept_contextual_embeddings))]
        
        for i in range(len(example_concepts) -1):
            concept1_name = example_concepts[i]
            concept2_name = example_concepts[i+1]
            emb1 = concept_contextual_embeddings[concept1_name].reshape(1, -1)
            emb2 = concept_contextual_embeddings[concept2_name].reshape(1, -1)
            
            sim = cosine_similarity(emb1, emb2)
            print(f"  ContextSim('{concept1_name}', '{concept2_name}'): {sim[0][0]:.4f}")

    return concept_contextual_embeddings

# 4. Learn Representations (Contrastive Learning) - Placeholder
def learn_representations_contrastive(all_concepts_data, image_folder_path=None, sentence_model=None, concept_field='noun_phrases'):
    """
    Expanded placeholder for learning concept/segment representations using contrastive learning.
    This function outlines the major steps and considerations. A full implementation
    requires a deep learning framework (PyTorch/TensorFlow) and a training pipeline.

    Args:
        all_concepts_data: List of dicts from merged data.
        image_folder_path: Path to the directory containing images (for image-similarity based pairs).
        sentence_model: A pre-trained SentenceTransformer model (can be used as a base or for encoding).
        concept_field: The key for concepts (e.g., 'noun_phrases') in all_concepts_data.
    """
    print("\n--- 4. Contrastive Learning for Representations (Detailed Outline) ---")
    print("This section outlines the approach for Option A: Contrastive Learning for Phrases/Sentences.")
    print("A full implementation requires a DL framework (PyTorch/TensorFlow), a training loop, and careful data handling.")

    if not sentence_model:
        print("Warning: A sentence_model (SentenceTransformer) is highly recommended as a starting point or for encoding text units. Proceeding with outline.")

    # --- I. Required Libraries (Illustrative) ---
    print("\nI. Potential Libraries:")
    print("   - PyTorch or TensorFlow: For building and training the neural network.")
    print("   - Hugging Face Transformers: For Transformer models if not using SentenceTransformer's built-in fine-tuning.")
    print("   - torchvision (if using image-based positive pairs): For image loading and pre-trained CNNs.")
    print("   - NLTK/spaCy: For advanced text processing if needed (e.g., sentence tokenization is already used elsewhere).")

    # --- II. Data Preparation: Defining Text Units and Pairs ---
    print("\nII. Data Preparation:")
    
    # A. Define Text Units for Learning
    # These are the items whose representations we want to learn.
    # Option 1: The concepts themselves (e.g., noun phrases).
    # Option 2: Sentences containing these concepts (provides more context).
    # Let's assume Option 2 for this outline.
    
    text_units = [] # List to store (text_unit_string, source_image_id, source_description_idx)
    concept_to_sentence_texts_map = defaultdict(list) # Re-using logic from contextual similarity for mapping

    # (Similar to contextual_similarity, map concepts to sentences they appear in)
    print("   A. Identifying Text Units (Sentences containing concepts):")
    # This part would be similar to the beginning of `calculate_contextual_similarity`
    # to get `concept_to_sentence_texts_map` or directly a list of relevant sentences.
    # For brevity, we'll assume we have a way to get sentences associated with concepts.
    
    # Example: Iterate through all_concepts_data to extract sentences containing concepts
    # This is a simplified illustration.
    all_sentences_with_source_info = [] # list of (sentence_text, image_id, description_index)
    for idx, item in enumerate(all_concepts_data):
        original_reason = item.get('original_reason', "")
        if isinstance(original_reason, list): original_reason = " ".join(original_reason)
        if not original_reason.strip(): continue
        
        current_concepts = item.get(concept_field, [])
        if not current_concepts: continue
            
        item_sentences = nltk.sent_tokenize(original_reason) # Ensure punkt is available
        
        for concept_text in set(c.strip() for c in current_concepts if isinstance(c,str) and c.strip()):
            for sent in item_sentences:
                if concept_text.lower() in sent.lower():
                    # Add sentence, its original image_id, and an index for the description
                    all_sentences_with_source_info.append({'text': sent, 'image_id': item.get('image_id'), 'description_idx': idx, 'concept_example': concept_text})
    
    if not all_sentences_with_source_info:
        print("     No sentences containing concepts found. Cannot proceed with contrastive learning outline.")
        return None
    print(f"     Extracted {len(all_sentences_with_source_info)} potential text units (sentences with concepts).")


    print("\n   B. Positive Pair Generation Strategies:")
    # Positive pairs are (anchor_text_unit, positive_text_unit)

    # Strategy B.1: Intra-Description Pairs
    # Sentences from the SAME VLM description (containing different or same concepts)
    # Rationale: Text units from the same explanation are likely related.
    positive_pairs_intra = []
    # Group sentences by their original description_idx
    sentences_by_description = defaultdict(list)
    for sent_info in all_sentences_with_source_info:
        sentences_by_description[sent_info['description_idx']].append(sent_info['text'])
    
    for desc_idx, sents_in_desc in sentences_by_description.items():
        unique_sents_in_desc = sorted(list(set(sents_in_desc))) # Unique sentences per description
        if len(unique_sents_in_desc) > 1:
            for i in range(len(unique_sents_in_desc)):
                for j in range(i + 1, len(unique_sents_in_desc)):
                    positive_pairs_intra.append((unique_sents_in_desc[i], unique_sents_in_desc[j]))
    print(f"     Generated {len(positive_pairs_intra)} intra-description positive pairs (sentence-sentence).")

    # Strategy B.2: Inter-Description (Image-Similarity Based - Advanced)
    # Text units from descriptions of images with HIGHLY SIMILAR IMAGE FEATURES.
    # This is complex and requires an image processing pipeline.
    print("     Strategy B.2: Inter-Description (Image Similarity) - Outline:")
    print(f"       - Requires image embeddings for images in: {image_folder_path if image_folder_path else 'Not Provided'}")
    print("       - Steps: ")
    print("         1. Load images (e.g., using PIL/OpenCV and `torchvision.transforms`).")
    print("         2. Extract features using a pre-trained CNN (e.g., ResNet from `torchvision.models`).")
    print("         3. Calculate pairwise image similarities (e.g., cosine similarity of features).")
    print("         4. Identify pairs of highly similar images.")
    print("         5. Form positive pairs from text units (sentences/concepts) associated with these similar image pairs.")
    # positive_pairs_image_sim = [] # Would be populated here

    # Strategy B.3: Paraphrasing (Optional)
    # Synthetically generate paraphrases of text units.
    print("     Strategy B.3: Paraphrasing - Outline:")
    print("       - Requires a paraphrase generation model (e.g., T5-based).")
    print("       - For each text unit, generate one or more paraphrases to form positive pairs.")
    # positive_pairs_paraphrase = []

    # Combine all positive pairs (example for intra-description only for now)
    all_positive_pairs = positive_pairs_intra 
    if not all_positive_pairs:
        print("     No positive pairs generated. Contrastive learning setup cannot proceed effectively.")
        # return None # Or continue with a different strategy if available

    print("\n   C. Negative Pair Generation:")
    print("     - In-Batch Negatives: For an (anchor, positive) pair, other samples in the same batch serve as negatives.")
    print("     - Hard Negatives (Advanced): Mining for samples that are semantically similar but not true positives (requires care).")

    # --- III. Model Architecture ---
    print("\nIII. Model Architecture:")
    print("   - Base Encoder: Use the provided `sentence_model` (e.g., SentenceTransformer).")
    print("   - Fine-Tuning: The goal is to fine-tune this encoder.")
    print("   - Projection Head (Optional but common): Add a small MLP on top of the encoder's output before loss calculation.")
    # Example:
    # if sentence_model is not None and hasattr(sentence_model, 'parameters'): # Check if it's a PyTorch model
    #     for param in sentence_model.parameters():
    #         param.requires_grad = True # Ensure model is trainable

    # --- IV. Loss Function ---
    print("\nIV. Loss Function: InfoNCE (Noise Contrastive Estimation)")
    print("   - Formula sketch: loss = -log [ exp(sim(anchor, positive) / temp) / sum_i(exp(sim(anchor, negative_i) / temp)) ]")
    print("   - `sim` is a similarity function (e.g., cosine similarity).")
    print("   - `temp` is a temperature hyperparameter.")

    # --- V. Training Loop (Conceptual Sketch) ---
    print("\nV. Training Loop (Conceptual Sketch - PyTorch-like):")
    print("   1. Prepare DataLoader for `all_positive_pairs` (and potentially pre-sampled negatives or use in-batch).")
    print("      - Each batch might yield multiple (anchor, positive, [negative_list]) sets.")
    print("   2. Initialize optimizer (e.g., AdamW).")
    print("   3. For each epoch:")
    print("      For each batch:")
    print("         a. Get anchor embeddings, positive embeddings, negative embeddings using the `sentence_model` (encoder).")
    print("         b. Calculate InfoNCE loss.")
    print("         c. Backpropagate loss and update model weights (optimizer.step()).")
    print("   4. Save the fine-tuned model.")
    
    print("\nThis detailed outline provides a roadmap. Actual implementation is a substantial task.")
    return None # Placeholder return


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting concept analysis script...")
    # Add NLTK resource check, similar to c_text_preprocess.py
    # Required for nltk.sent_tokenize used in contextual_similarity
    nltk_resources_to_check = {
        'punkt': 'tokenizers/punkt',
        # Add other NLTK resources if other NLTK functionalities are added directly to this script
    }
    print("Checking NLTK resources for concept_analysis.py...")
    all_nltk_available = True
    for resource_name, resource_path in nltk_resources_to_check.items():
        try:
            nltk.data.find(resource_path)
            print(f" - NLTK Resource '{resource_name}' already available.")
        except LookupError:
            print(f" - NLTK Resource '{resource_name}' not found. Attempting download...")
            try:
                nltk.download(resource_name, quiet=False)
                nltk.data.find(resource_path) # Verify after download
                print(f"   Successfully downloaded and verified '{resource_name}'.")
            except Exception as e:
                print(f"   Failed to download or verify '{resource_name}'. Error: {e}")
                all_nltk_available = False
        except Exception as e: # Catch any other unexpected errors during find
            print(f"   An unexpected error occurred while checking for NLTK resource '{resource_name}': {e}")
            all_nltk_available = False
    
    if not all_nltk_available:
        print("One or more NLTK resources required by concept_analysis.py could not be made available. Some functionalities might fail.")
        # Decide if to exit: exit(1) 
    else:
        print("All checked NLTK resources for concept_analysis.py are available.\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths
    processed_concepts_path = os.path.join(script_dir, PROCESSED_CONCEPTS_FILE)
    val_csv_path = os.path.join(script_dir, VAL_CSV_FILE)
    
    # If VAL_CSV_FILE was set to the example ISIC path, it will be absolute
    # If it's just "val.csv", os.path.join(script_dir, "val.csv") makes it absolute.

    print(f"Attempting to load processed concepts from: {processed_concepts_path}")
    print(f"Attempting to load ground truth from: {val_csv_path}")
    if not os.path.exists(val_csv_path):
        print(f"Warning: {VAL_CSV_FILE} not found at {val_csv_path}. Please ensure the file exists or update VAL_CSV_FILE path.")


    # Load data
    concepts_data_list = load_processed_concepts(processed_concepts_path)
    if concepts_data_list is None: # Indicates a fatal error during loading
        print("Exiting due to failure in loading processed concepts.")
        exit(1)
    if not concepts_data_list: # Empty list after successful load (e.g. file was empty or all entries invalid)
        print("No valid concept data loaded. Exiting.")
        exit(1)

    truth_data_df = load_ground_truth(val_csv_path)
    # truth_data_df can be None if file not found or error; merge_data handles this.

    # Merge ground truth with concept data
    all_data = merge_data(concepts_data_list, truth_data_df)

    if not all_data:
        print("No data available after loading and potential merging. Exiting.")
        exit(1)
    
    # print(f"\nFirst example of merged data entry:\n{json.dumps(all_data[0], indent=2)}")

    # --- 1. Semantic Similarity (using Noun Phrases) ---
    print("\n--- 1. Semantic Similarity (p_theta(j|i) using Noun Phrase Embeddings) ---")
    
    # Extract all unique, non-empty noun phrases from the merged data
    all_noun_phrases_set = set()
    for item in all_data:
        # item['noun_phrases'] should be a list of strings
        phrases = item.get("noun_phrases", []) 
        if isinstance(phrases, list):
            for phrase in phrases:
                if isinstance(phrase, str) and phrase.strip():
                    all_noun_phrases_set.add(phrase.strip())
    
    unique_noun_phrases_list = sorted(list(all_noun_phrases_set))
    
    st_model = None # Initialize sentence transformer model variable
    concept_embeddings_np = np.array([])

    if not unique_noun_phrases_list:
        print("No unique noun phrases found in the data to analyze for semantic similarity.")
    else:
        print(f"Found {len(unique_noun_phrases_list)} unique noun phrases for similarity analysis.")
        if len(unique_noun_phrases_list) > 10000:
             print("Warning: A large number of unique noun phrases (>10,000). Embedding generation might be slow and consume significant memory.")

        # Dynamically import SentenceTransformer and load model
        try:
            from sentence_transformers import SentenceTransformer
            print("Attempting to load SentenceTransformer model (e.g., 'all-MiniLM-L6-v2')...")
            print("(This may take a moment, especially on the first run as the model might be downloaded.)")
            # Common model, good balance of speed and performance.
            # For more domain-specific (biomedical), consider models like:
            # 'allenai/scibert_scivocab_uncased' (requires `transformers` library too)
            # 'dmis-lab/biobert-base-cased-v1.1'
            # Or sentence versions if available on SentenceTransformers.
            st_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully.")
        except ImportError:
            print("\nError: The 'sentence-transformers' library is not installed.")
            print("Please install it by running: pip install sentence-transformers")
            print("Semantic similarity analysis will be skipped.\n")
        except Exception as e: # Catch other errors during model loading (e.g., network issues)
            print(f"\nError loading SentenceTransformer model: {e}")
            print("Please ensure 'sentence-transformers' is installed correctly, you have an internet connection (for model download),")
            print("and the model name is valid. Semantic similarity analysis will be skipped.\n")
            
        if st_model:
            concept_embeddings_np = get_concept_embeddings(unique_noun_phrases_list, st_model)

            if concept_embeddings_np.ndim == 2 and concept_embeddings_np.shape[0] > 0:
                print(f"Generated embeddings of shape: {concept_embeddings_np.shape} for {len(unique_noun_phrases_list)} phrases.")

                # Show similarity for a few example pairs
                if len(unique_noun_phrases_list) >= 2:
                    print("\nExample semantic similarities (cosine similarity of embeddings):")
                    num_examples = min(5, len(unique_noun_phrases_list) - 1)
                    for i in range(num_examples):
                        sim = cosine_similarity(concept_embeddings_np[i].reshape(1, -1), 
                                                concept_embeddings_np[i+1].reshape(1, -1))
                        print(f"  Sim('{unique_noun_phrases_list[i]}', '{unique_noun_phrases_list[i+1]}'): {sim[0][0]:.4f}")
                
                # Demonstrate finding neighbors for the first concept in the list
                if unique_noun_phrases_list:
                    target_concept_example = "melanoma" # Changed to target "melanoma"
                    print(f"\nFinding top 5 neighbors for concept: '{target_concept_example}'")
                    
                    try:
                        target_index = unique_noun_phrases_list.index(target_concept_example)
                        target_embedding_example = concept_embeddings_np[target_index]
                        
                        neighbors = calculate_semantic_similarity_for_concept(
                            target_concept_example, 
                            target_embedding_example, 
                            unique_noun_phrases_list, 
                            concept_embeddings_np, 
                            top_n=5
                        )
                        if neighbors:
                            for neighbor, score in neighbors:
                                print(f"  - '{neighbor}' (Similarity: {score:.4f})")
                        else:
                            # This case might occur if "melanoma" is the only concept or other specific conditions
                            print(f"No other concepts found to compare with '{target_concept_example}' or it's the only concept.")
                    except ValueError:
                        print(f"Concept '{target_concept_example}' not found in the list of unique noun phrases. Cannot find neighbors.")
                    except IndexError:
                        print(f"Could not retrieve embedding for '{target_concept_example}'. Index out of bounds.")
            else:
                print("Failed to generate valid concept embeddings for noun phrases.")
        else: # st_model is None (due to import error or loading error)
            print("Skipping semantic similarity calculations as the SentenceTransformer model could not be loaded.")

    # --- 2. Co-occurrence within Descriptions ---
    # This uses the 'noun_phrases' field by default.
    cooccurrence_df = calculate_cooccurrence(all_data, concept_field='noun_phrases')
    if cooccurrence_df is not None and not cooccurrence_df.empty:
        print("\nExample of Co-occurrence Matrix (Top 5x5, if available):")
        # To make it more readable, show a small slice.
        # Displaying a large matrix isn't practical here.
        slice_size = min(5, cooccurrence_df.shape[0], cooccurrence_df.shape[1])
        if slice_size > 0:
            print(cooccurrence_df.iloc[:slice_size, :slice_size])
            # Example: How to find co-occurrence for a specific pair
            if len(cooccurrence_df) > 1 and 'typical nevus morphology' in cooccurrence_df.index and 'uniformity' in cooccurrence_df.columns :
                 print(f"Co-occurrence of 'typical nevus morphology' and 'uniformity': {cooccurrence_df.loc['typical nevus morphology', 'uniformity']}")
        else:
            print("Co-occurrence matrix was generated but is too small to display a 5x5 slice.")
    elif cooccurrence_df is None:
        print("Co-occurrence calculation was skipped or failed.")
    else: # Empty DataFrame
        print("Co-occurrence matrix is empty (no co-occurring concepts found or processed).")


    # --- 3. Contextual Similarity ---
    # Pass the loaded sentence model if available
    contextual_embeddings = calculate_contextual_similarity(all_data, st_model, concept_field='noun_phrases')
    if contextual_embeddings:
        print(f"\nGenerated {len(contextual_embeddings)} contextual embeddings for concepts.")
        # Further analysis can be done with these contextual_embeddings
    else:
        print("\nContextual similarity calculation was skipped or did not yield results.")

    # --- 4. Learn Representations (Contrastive Learning - Placeholder Call) ---
    # Pass the sentence model for encoding, and image path for image-based strategies
    learn_representations_contrastive(all_data, image_folder_path=IMAGE_TRAINING_DIR, sentence_model=st_model, concept_field='noun_phrases')

    print("\nConcept analysis script finished.")

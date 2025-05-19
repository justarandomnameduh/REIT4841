# Use TFIDF and KMeans for clustering

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings

# Suppress specific warnings from sklearn KMeans
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")


def load_predictions(filepath):
    """Loads predictions from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def extract_reason_texts(predictions_data):
    """Extracts and preprocesses unique reason texts from predictions data."""
    all_texts = []
    if not predictions_data:
        return []

    for item in predictions_data:
        if "reason" in item and isinstance(item["reason"], list):
            for text_element in item["reason"]:
                processed_text = text_element.strip().lower()
                # Filter out generic/error reasons and very short texts
                if processed_text and \
                   processed_text not in ["general appearance", "error: image file not found", "image file not found at path"] and \
                   not processed_text.startswith("error during processing:") and \
                   len(processed_text.split()) > 2: # Ensure text has more than two words
                    all_texts.append(processed_text)
    return list(set(all_texts)) # Return unique texts

def cluster_texts_and_extract_keywords(texts, num_clusters=10, num_top_keywords_per_concept=5):
    """Clusters texts using TF-IDF and K-Means, then identifies top keywords for each cluster."""
    if not texts:
        print("No texts to cluster.")
        return {}

    effective_num_clusters = min(num_clusters, len(texts))

    if effective_num_clusters == 0:
        print("No texts available for clustering after filtering.")
        return {}
    
    if effective_num_clusters == 1 and len(texts) >= 1:
         print("Only one effective cluster can be formed. All texts will be in one group.")
         vectorizer_single = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1, ngram_range=(1, 3))
         try:
            tfidf_matrix_single = vectorizer_single.fit_transform(texts)
            if tfidf_matrix_single.shape[1] > 0:
                terms_single = vectorizer_single.get_feature_names_out()
                summed_tfidf = tfidf_matrix_single.sum(axis=0).A1 
                top_term_indices = summed_tfidf.argsort()[-num_top_keywords_per_concept:][::-1]
                top_keywords = [terms_single[idx] for idx in top_term_indices]
                return {0: {"concept_keywords": top_keywords, "member_texts": texts}}
            else:
                return {0: {"concept_keywords": ["No keywords extractable"], "member_texts": texts}}
         except ValueError:
            return {0: {"concept_keywords": ["Error in keyword extraction"], "member_texts": texts}}

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=max(1, int(len(texts)*0.05)), ngram_range=(1, 3))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        print(f"Error during TF-IDF vectorization (e.g., all texts are stop words or too short): {e}")
        return {0: {"concept_keywords": ["TF-IDF Error"], "member_texts": texts}}

    if tfidf_matrix.shape[1] == 0:
        print("TF-IDF matrix has no features (e.g., all terms filtered out). Cannot perform clustering.")
        return {0: {"concept_keywords": ["No features for clustering"], "member_texts": texts}}
    
    actual_n_clusters = min(effective_num_clusters, tfidf_matrix.shape[0])
    if actual_n_clusters < 1:
        print("Cannot form clusters with less than 1 sample.")
        return {0: {"concept_keywords": ["Clustering error: <1 sample"], "member_texts": texts}}
    if actual_n_clusters == 1 and tfidf_matrix.shape[0] >= 1:
        terms_single = vectorizer.get_feature_names_out()
        summed_tfidf = tfidf_matrix.sum(axis=0).A1
        top_term_indices = summed_tfidf.argsort()[-num_top_keywords_per_concept:][::-1]
        top_keywords = [terms_single[idx] for idx in top_term_indices]
        return {0: {"concept_keywords": top_keywords, "member_texts": texts}}

    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
    
    try:
        kmeans.fit(tfidf_matrix)
    except Exception as e:
        print(f"Error during K-Means fitting: {e}")
        return {0: {"concept_keywords": ["K-Means Error"], "member_texts": texts}}

    clustered_data = {i: {"concept_keywords": [], "member_texts": []} for i in range(actual_n_clusters)}
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    all_labels = kmeans.labels_
    for i in range(actual_n_clusters):
        members = [texts[j] for j, label_idx in enumerate(all_labels) if label_idx == i]
        clustered_data[i]["member_texts"] = members
        
        current_keywords = []
        if i < order_centroids.shape[0]:
            for term_idx_pos, term_idx in enumerate(order_centroids[i, :]):
                if len(current_keywords) >= num_top_keywords_per_concept:
                    break
                if term_idx < len(terms):
                    current_keywords.append(terms[term_idx])
        clustered_data[i]["concept_keywords"] = current_keywords if current_keywords else ["No specific keywords"]
            
    return clustered_data

def main():
    predictions_filepath = "predictions_mel_nv.json"
    output_concepts_filepath = "extracted_concepts.json"

    if not os.path.exists(predictions_filepath):
        print(f"Warning: Predictions file '{predictions_filepath}' not found.")
        return

    predictions_data = load_predictions(predictions_filepath)
    if not predictions_data:
        print("Could not load or process predictions data. Exiting.")
        return

    reason_texts = extract_reason_texts(predictions_data)
    if not reason_texts:
        print("No valid reason texts extracted to cluster. Exiting.")
        return
    
    print(f"Extracted {len(reason_texts)} unique reason texts for clustering.")

    if len(reason_texts) < 5:
        num_concepts_to_extract = 1
    elif len(reason_texts) < 20:
        num_concepts_to_extract = max(1, len(reason_texts) // 3)
    else:
        num_concepts_to_extract = min(10, max(1, len(reason_texts) // 5))
    
    print(f"Attempting to find {num_concepts_to_extract} concepts.")
    
    clustered_concepts = cluster_texts_and_extract_keywords(
        reason_texts, 
        num_clusters=num_concepts_to_extract,
        num_top_keywords_per_concept=5
    )
    
    if not clustered_concepts:
        print("No concepts were extracted.")
        return

    print("\n--- Extracted Concepts ---")
    for cluster_id, data in clustered_concepts.items():
        print(f"\nConcept Cluster {cluster_id}:")
        if data.get('concept_keywords'):
            print(f"  Representative Keywords: {'; '.join(data['concept_keywords'])}")
        else:
            print("  Representative Keywords: None identified")
        
    try:
        with open(output_concepts_filepath, 'w') as f:
            json.dump(clustered_concepts, f, indent=2)
        print(f"\nClustered concepts saved to {output_concepts_filepath}")
    except IOError:
        print(f"Error: Could not write concepts to {output_concepts_filepath}")

if __name__ == "__main__":
    main()

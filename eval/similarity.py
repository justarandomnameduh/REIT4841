import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk import word_tokenize, ngrams
from tqdm import tqdm

# Download nltk data
nltk.download('punkt', quiet=True)

# Paths
clusters_path = "/home/nqmtien/REIT4841/pipeline/concept_extractor/clusters/gemini/ham10000"
concepts_path = "/home/nqmtien/REIT4841/pipeline/concept_extractor/concepts/gemini/ham10000"

# Values
k_values = [1, 3, 5, 10]
n_values = [1, 2, 4]
c_values = [1, 3, 5, 10, 30, 50, 100, 200]

# Load model (BioBERT)
model_name = 'dmis-lab/biobert-v1.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Cache for embeddings
embedding_cache = {}

def encode_text(text):
    if text in embedding_cache:
        return embedding_cache[text]
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embedding = embeddings.squeeze()
    embedding_cache[text] = embedding
    return embedding

def get_phrases(text):
    tokens = word_tokenize(text)
    phrases = set(tokens)  # unigrams
    phrases.update(' '.join(bg) for bg in ngrams(tokens, 2))  # bigrams
    # phrases.update(' '.join(tg) for tg in ngrams(tokens, 3))  # trigrams
    # phrases.update(' '.join(tg) for tg in ngrams(tokens, 4))  # quadgrams

    return list(phrases)

# For each k-n pair
for k in tqdm(k_values, desc="Processing k values"):
    for n in tqdm(n_values, desc="Processing n values", leave=False):
        # Load paragraphs
        paragraph_file = f"paragraph_{k}_{n}.json"
        paragraph_filepath = os.path.join(clusters_path, paragraph_file)
        with open(paragraph_filepath, 'r') as f:
            paragraphs_data = json.load(f)
        paragraphs = [item['paragraph'] for item in paragraphs_data]
        
        # Dictionary to hold similarities for each c
        similarities = {}
        
        # For each c
        for c in tqdm(c_values, desc="Processing c values", leave=False):
            concept_file = f"concept_all_{k}_{n}_{c}.json"
            concept_filepath = os.path.join(concepts_path, concept_file)
            with open(concept_filepath, 'r') as f:
                concept_data = json.load(f)
            concepts = concept_data['all']
            
            # Compute max similarities for each concept
            sims = []
            for conc in tqdm(concepts, desc="Processing concepts", leave=False):
                conc_emb = encode_text(conc)
                max_sim = 0
                for para in paragraphs:
                    phrases = get_phrases(para)
                    for phrase in phrases:
                        phrase_emb = encode_text(phrase)
                        sim = torch.cosine_similarity(phrase_emb.unsqueeze(0), conc_emb.unsqueeze(0)).item()
                        if sim > max_sim:
                            max_sim = sim
                sims.append(max_sim)
            similarities[c] = sims
        
        # Save similarities data as JSON
        json_filename = f'similarity_data_k{k}_n{n}.json'
        json_data = {
            'k': k,
            'n': n,
            'c_values': c_values,
            'similarities': {str(c): similarities[c] for c in c_values}
        }
        with open(json_filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        print(f"Saved data for k={k}, n={n} as {json_filename}")
        
        # Plot box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [similarities[c] for c in c_values]
        ax.boxplot(data_to_plot)
        ax.set_xticks(range(1, len(c_values) + 1))
        ax.set_xticklabels([str(c) for c in c_values])
        ax.set_xlabel('c')
        ax.set_ylabel('Similarity Score')
        ax.set_title(f'Similarity Distribution for k={k}, n={n}')
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        plot_filename = f'similarity_k{k}_n{n}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot for k={k}, n={n} as {plot_filename}")
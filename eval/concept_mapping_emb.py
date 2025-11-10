import os
import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pandas as pd


class ClinicalRelevanceEvaluator:
    def __init__(self):
        """Initialize BioBERT model for semantic similarity."""
        print("Loading BioBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Cache for embeddings
        self.embedding_cache = {}
    
    def get_embedding(self, text):
        """Get BioBERT embedding for a text (with caching)."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embedding = embedding.flatten()
        self.embedding_cache[text] = embedding
        return embedding
    
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity (normalized dot product) between two embeddings."""
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        # Compute dot product
        return np.dot(emb1_norm, emb2_norm)
    
    def load_target_vocabularies(self, dataset, use_full_icd10=False):
        """
        Load the three target vocabularies:
        1. Derm7pt concepts (human-annotated)
        2. Dermlike concepts (dermatological features from literature)
        3. ICD-10-CM concepts (standardized medical codes)
        
        Args:
            dataset: Dataset name (e.g., 'ham10000')
            use_full_icd10: If True, use full ICD-10-CM (~29K), else use 30-sample
        """
        base_path = Path(__file__).parent.parent
        manual_concepts_dir = base_path / 'eval' / 'manual_concepts'
        general_concepts_dir = base_path / 'eval' / 'general_concepts'
        
        vocabularies = {}
        
        # Load Derm7pt
        derm7pt_file = manual_concepts_dir / f'{dataset}_derm7pt.json'
        if derm7pt_file.exists():
            with open(derm7pt_file, 'r') as f:
                data = json.load(f)
                vocabularies['derm7pt'] = data['all']
                print(f"Loaded {len(vocabularies['derm7pt'])} Derm7pt concepts")
        else:
            print(f"Warning: Derm7pt file not found: {derm7pt_file}")
            vocabularies['derm7pt'] = []
        
        # Load Dermlike
        dermlike_file = manual_concepts_dir / f'{dataset}_dermlike.json'
        if dermlike_file.exists():
            with open(dermlike_file, 'r') as f:
                data = json.load(f)
                vocabularies['dermlike'] = data['all']
                print(f"Loaded {len(vocabularies['dermlike'])} Dermlike concepts")
        else:
            print(f"Warning: Dermlike file not found: {dermlike_file}")
            vocabularies['dermlike'] = []
        
        # Load ICD-10-CM
        # Use 30-sample version for faster processing (default), or full version if requested
        icd10_file_30 = general_concepts_dir / 'icd10cm_descriptions_30.txt'
        icd10_file_full = general_concepts_dir / 'icd10cm_descriptions.txt'
        
        if use_full_icd10:
            # User explicitly requested full vocabulary
            if icd10_file_full.exists():
                icd10_file = icd10_file_full
                print(f"Using FULL ICD-10-CM vocabulary (as requested)")
            else:
                print(f"Warning: Full ICD-10-CM not found, falling back to 30-sample")
                icd10_file = icd10_file_30 if icd10_file_30.exists() else None
        else:
            # Default: prefer 30-sample for efficiency
            if icd10_file_30.exists():
                icd10_file = icd10_file_30
                print(f"Using ICD-10-CM 30-sample for efficiency (use --use-full-icd10 for full vocabulary)")
            elif icd10_file_full.exists():
                icd10_file = icd10_file_full
                print(f"30-sample not found, using full ICD-10-CM vocabulary")
            else:
                icd10_file = None
        
        if icd10_file and icd10_file.exists():
            with open(icd10_file, 'r') as f:
                vocabularies['icd10cm'] = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(vocabularies['icd10cm'])} ICD-10-CM concepts from {icd10_file.name}")
        else:
            print(f"Warning: ICD-10-CM file not found")
            vocabularies['icd10cm'] = []
        
        return vocabularies
    
    def compute_vocabulary_embeddings(self, vocabularies):
        """Pre-compute embeddings for all vocabulary concepts."""
        vocab_embeddings = {}
        
        for vocab_name, concepts in vocabularies.items():
            print(f"\nComputing embeddings for {vocab_name} ({len(concepts)} concepts)...")
            embeddings = {}
            for concept in tqdm(concepts, desc=vocab_name):
                embeddings[concept] = self.get_embedding(concept)
            vocab_embeddings[vocab_name] = embeddings
        
        return vocab_embeddings
    
    def compute_inter_concept_distances(self, vocab_embeddings):
        """
        Compute inter-concept distances within each vocabulary to determine
        how distinct/specific the concepts are.
        
        Returns dictionary with statistics for each vocabulary.
        """
        print("\n" + "="*80)
        print("Computing inter-concept distances (within vocabularies)...")
        print("="*80)
        
        inter_distances = {}
        min_similarities = {}  # Store minimum similarity for each vocabulary
        max_similarities = {}  # Store maximum similarity for each vocabulary
        
        for vocab_name, embeddings in vocab_embeddings.items():
            concepts = list(embeddings.keys())
            similarities = []
            
            print(f"\n{vocab_name}: Computing {len(concepts)}x{len(concepts)} similarities...")
            
            for i, concept_i in enumerate(concepts):
                for j, concept_j in enumerate(concepts):
                    if i < j:  # Only compute upper triangle
                        sim = self.compute_similarity(
                            embeddings[concept_i],
                            embeddings[concept_j]
                        )
                        similarities.append(sim)
            
            similarities = np.array(similarities)
            
            # Calculate statistics
            stats = {
                'mean': float(np.mean(similarities)),
                'median': float(np.median(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'q25': float(np.percentile(similarities, 25)),
                'q75': float(np.percentile(similarities, 75)),
                'num_pairs': len(similarities)
            }
            
            min_similarities[vocab_name] = stats['min']
            max_similarities[vocab_name] = stats['max']
            inter_distances[vocab_name] = stats
            
            print(f"  Mean inter-similarity: {stats['mean']:.4f}")
            print(f"  Std dev: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Calculate automatic thresholds based on clinical vocabularies
        if 'derm7pt' in min_similarities and 'dermlike' in min_similarities:
            # Minimum threshold (widest semantic gap - most lenient)
            min_threshold = (min_similarities['derm7pt'] + min_similarities['dermlike']) / 2.0
            
            # Maximum threshold (narrowest semantic gap - most strict)
            max_threshold = (max_similarities['derm7pt'] + max_similarities['dermlike']) / 2.0
            
            # Mid-range threshold (balance between min and max)
            global_auto_threshold = (min_threshold + max_threshold) / 2.0
            
            print(f"\n" + "="*80)
            print(f"Global Auto Threshold Calculation:")
            print(f"  Derm7pt min similarity: {min_similarities['derm7pt']:.4f}")
            print(f"  Dermlike min similarity: {min_similarities['dermlike']:.4f}")
            print(f"  Min threshold (most lenient): {min_threshold:.4f}")
            print(f"")
            print(f"  Derm7pt max similarity: {max_similarities['derm7pt']:.4f}")
            print(f"  Dermlike max similarity: {max_similarities['dermlike']:.4f}")
            print(f"  Max threshold (most strict): {max_threshold:.4f}")
            print(f"")
            print(f"  Global auto threshold (mid-range): {global_auto_threshold:.4f}")
            print("="*80)
        else:
            # Fallback if vocabularies not found
            global_auto_threshold = 0.70
            print(f"\nWarning: Could not compute global threshold, using default: {global_auto_threshold:.4f}")
        
        # Assign the same global threshold to all vocabularies
        for vocab_name in inter_distances:
            inter_distances[vocab_name]['auto_threshold'] = float(global_auto_threshold)
        
        return inter_distances
    
    def compute_cross_vocabulary_similarity(self, vocab_embeddings):
        """
        Compute similarity between different vocabularies to test if
        generated concepts are more similar to clinical concepts than
        clinical concepts are to each other.
        """
        print("\n" + "="*80)
        print("Computing cross-vocabulary similarities...")
        print("="*80)
        
        vocab_names = list(vocab_embeddings.keys())
        cross_similarities = {}
        
        for i, vocab1 in enumerate(vocab_names):
            for j, vocab2 in enumerate(vocab_names):
                if i < j:  # Only compute upper triangle
                    pair_key = f"{vocab1}_vs_{vocab2}"
                    print(f"\n{pair_key}:")
                    
                    concepts1 = list(vocab_embeddings[vocab1].keys())
                    concepts2 = list(vocab_embeddings[vocab2].keys())
                    
                    similarities = []
                    for c1 in concepts1:
                        max_sim = 0
                        for c2 in concepts2:
                            sim = self.compute_similarity(
                                vocab_embeddings[vocab1][c1],
                                vocab_embeddings[vocab2][c2]
                            )
                            max_sim = max(max_sim, sim)
                        similarities.append(max_sim)
                    
                    similarities = np.array(similarities)
                    
                    stats = {
                        'mean': float(np.mean(similarities)),
                        'median': float(np.median(similarities)),
                        'std': float(np.std(similarities)),
                        'min': float(np.min(similarities)),
                        'max': float(np.max(similarities))
                    }
                    
                    cross_similarities[pair_key] = stats
                    
                    print(f"  Mean max similarity: {stats['mean']:.4f}")
                    print(f"  Median: {stats['median']:.4f}")
        
        return cross_similarities
    
    def evaluate_concept_file(self, gen_file, vocabularies, vocab_embeddings, inter_distances, 
                              use_auto_threshold=True, fixed_threshold=0.7):
        """
        Evaluate a single generated concept file against all target vocabularies.
        Uses automatic thresholds based on inter-concept distances or fixed threshold.
        
        Returns comprehensive similarity statistics and mappings.
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {gen_file.name}")
        
        # Load generated concepts
        with open(gen_file, 'r') as f:
            gen_data = json.load(f)
        
        generated_concepts = gen_data['all']
        print(f"Generated concepts: {len(generated_concepts)}")
        
        # Compute embeddings for generated concepts
        print("\nComputing embeddings for generated concepts...")
        gen_embeddings = {}
        for concept in tqdm(generated_concepts):
            gen_embeddings[concept] = self.get_embedding(concept)
        
        # Evaluate against each vocabulary
        results = {}
        
        for vocab_name in vocabularies.keys():
            print(f"\nEvaluating against {vocab_name}...")
            
            vocab_concepts = vocabularies[vocab_name]
            vocab_emb = vocab_embeddings[vocab_name]
            
            # Determine threshold for this vocabulary
            if use_auto_threshold:
                threshold = inter_distances[vocab_name]['auto_threshold']
                print(f"  Using auto threshold: {threshold:.4f}")
            else:
                threshold = fixed_threshold
                print(f"  Using fixed threshold: {threshold:.4f}")
            
            # Compute all similarities
            all_similarities_matrix = []  # For specificity analysis
            max_similarities = []
            mappings = {}
            
            for gen_concept, gen_emb in tqdm(gen_embeddings.items(), desc=f"Similarities to {vocab_name}"):
                concept_sims = []
                matches = []
                
                for vocab_concept, vocab_emb_vec in vocab_emb.items():
                    sim = self.compute_similarity(gen_emb, vocab_emb_vec)
                    concept_sims.append(sim)
                    
                    if sim >= threshold:
                        matches.append({
                            'concept': vocab_concept,
                            'similarity': float(sim)
                        })
                
                # Sort matches by similarity
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                mappings[gen_concept] = matches
                
                # Store all similarities for this concept
                all_similarities_matrix.append(concept_sims)
                
                # Store max similarity for this generated concept
                if concept_sims:
                    max_similarities.append(max(concept_sims))
            
            # Compute statistics
            max_similarities = np.array(max_similarities)
            all_similarities_matrix = np.array(all_similarities_matrix)
            
            mapped_count = sum(1 for m in mappings.values() if m)
            total_mappings = sum(len(m) for m in mappings.values())
            
            # SPECIFICITY METRICS
            # 1. Average number of mappings per concept (lower = more specific)
            avg_mappings_per_concept = total_mappings / len(generated_concepts) if generated_concepts else 0
            
            # 2. Percentage mapping to few vs many concepts
            mapping_counts = [len(m) for m in mappings.values()]
            few_mappings = sum(1 for c in mapping_counts if 0 < c <= 3)  # Maps to 1-3 concepts
            many_mappings = sum(1 for c in mapping_counts if c > 5)  # Maps to >5 concepts
            
            # 3. Standard deviation of similarities (higher = more discriminative)
            std_per_concept = np.std(all_similarities_matrix, axis=1)
            mean_std = float(np.mean(std_per_concept))
            
            # 4. Similarity range per concept (max - min)
            ranges = np.max(all_similarities_matrix, axis=1) - np.min(all_similarities_matrix, axis=1)
            mean_range = float(np.mean(ranges))
            
            results[vocab_name] = {
                'mappings': mappings,
                'statistics': {
                    'vocab_size': len(vocab_concepts),
                    'threshold_used': threshold,
                    'threshold_type': 'auto' if use_auto_threshold else 'fixed',
                    # Similarity statistics
                    'mean_max_similarity': float(np.mean(max_similarities)) if len(max_similarities) > 0 else 0.0,
                    'median_max_similarity': float(np.median(max_similarities)) if len(max_similarities) > 0 else 0.0,
                    'std_max_similarity': float(np.std(max_similarities)) if len(max_similarities) > 0 else 0.0,
                    'min_max_similarity': float(np.min(max_similarities)) if len(max_similarities) > 0 else 0.0,
                    'max_max_similarity': float(np.max(max_similarities)) if len(max_similarities) > 0 else 0.0,
                    # Mapping statistics
                    'above_threshold': int(np.sum(max_similarities >= threshold)),
                    'mapped_concepts': mapped_count,
                    'total_mappings': total_mappings,
                    # Specificity metrics
                    'avg_mappings_per_concept': float(avg_mappings_per_concept),
                    'concepts_with_few_mappings': few_mappings,
                    'concepts_with_many_mappings': many_mappings,
                    'mean_std_across_vocab': mean_std,
                    'mean_similarity_range': mean_range,
                    'specificity_score': mean_std / (avg_mappings_per_concept + 1)  # Higher = more specific
                }
            }
            
            print(f"  Mean max similarity: {results[vocab_name]['statistics']['mean_max_similarity']:.4f}")
            print(f"  Median max similarity: {results[vocab_name]['statistics']['median_max_similarity']:.4f}")
            print(f"  Concepts above threshold: {results[vocab_name]['statistics']['above_threshold']}/{len(generated_concepts)}")
            print(f"  Avg mappings per concept: {avg_mappings_per_concept:.2f}")
            print(f"  Specificity score: {results[vocab_name]['statistics']['specificity_score']:.4f}")
        
        return {
            'generated_file': str(gen_file),
            'num_generated_concepts': len(generated_concepts),
            'use_auto_threshold': use_auto_threshold,
            'vocabulary_results': results
        }
    
    def save_results(self, result, output_file):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Clinical Relevance Evaluation using BioBERT embeddings'
    )
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Fixed similarity threshold (default: 0.7, only used if --no-auto-threshold)')
    parser.add_argument('--dataset', type=str, default='ham10000',
                       help='Dataset name (default: ham10000)')
    parser.add_argument('--no-auto-threshold', action='store_true',
                       help='Use fixed threshold instead of automatic thresholds')
    parser.add_argument('--use-full-icd10', action='store_true',
                       help='Use full ICD-10-CM vocabulary instead of 30-sample (slower)')
    
    args = parser.parse_args()
    use_auto_threshold = not args.no_auto_threshold
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    gen_concepts_dir = base_path / 'pipeline' / 'concept_extractor' / 'concepts' / 'gemini' / args.dataset
    output_dir = base_path / 'eval' / 'clinical_relevance' / args.dataset
    
    # Initialize evaluator
    evaluator = ClinicalRelevanceEvaluator()
    
    # Load target vocabularies
    print("\n" + "="*80)
    print("Loading target vocabularies...")
    print("="*80)
    vocabularies = evaluator.load_target_vocabularies(args.dataset, args.use_full_icd10)
    
    if not any(vocabularies.values()):
        print("Error: No vocabularies loaded")
        return
    
    # Pre-compute vocabulary embeddings
    print("\n" + "="*80)
    print("Pre-computing vocabulary embeddings...")
    print("="*80)
    vocab_embeddings = evaluator.compute_vocabulary_embeddings(vocabularies)
    
    # Compute inter-concept distances (for automatic thresholds)
    inter_distances = evaluator.compute_inter_concept_distances(vocab_embeddings)
    
    # Compute cross-vocabulary similarities (for Point 3)
    cross_vocab_sims = evaluator.compute_cross_vocabulary_similarity(vocab_embeddings)
    
    # Save inter-distances and cross-vocab similarities
    meta_output = {
        'inter_concept_distances': inter_distances,
        'cross_vocabulary_similarities': cross_vocab_sims,
        'threshold_mode': 'auto' if use_auto_threshold else 'fixed',
        'fixed_threshold': args.threshold if not use_auto_threshold else None
    }
    meta_file = output_dir / 'vocabulary_analysis.json'
    os.makedirs(os.path.dirname(meta_file), exist_ok=True)
    with open(meta_file, 'w') as f:
        json.dump(meta_output, f, indent=2)
    print(f"\nVocabulary analysis saved to: {meta_file}")
    
    # Define files to process: k=[3,5,10], n=[2,4], c=[50,100,200]
    # Plus additional specific configurations
    k_values = [3, 5, 10]
    n_values = [2, 4]
    c_values = [50, 100, 200]
    
    # Additional specific configurations requested
    additional_configs = [
        (3, 2, 5),
        (5, 4, 10)
    ]
    
    total_files = len(k_values) * len(n_values) * len(c_values) + len(additional_configs)
    print(f"\n{'='*80}")
    print(f"Processing {total_files} file combinations...")
    print(f"Standard grid: k={k_values}, n={n_values}, c={c_values}")
    print(f"Additional configs: {additional_configs}")
    print(f"Threshold mode: {'AUTO' if use_auto_threshold else 'FIXED'}")
    print("="*80)
    
    # Store all results for summary
    all_results = []
    processed = 0
    
    # Process standard grid
    for k in k_values:
        for n in n_values:
            for c in c_values:
                gen_file = gen_concepts_dir / f'concept_all_{k}_{n}_{c}.json'
                
                if not gen_file.exists():
                    print(f"\nWarning: Generated file not found: {gen_file}")
                    continue
                
                output_file = output_dir / f'evaluation_{k}_{n}_{c}.json'
                
                try:
                    result = evaluator.evaluate_concept_file(
                        gen_file, 
                        vocabularies, 
                        vocab_embeddings,
                        inter_distances,
                        use_auto_threshold,
                        args.threshold
                    )
                    evaluator.save_results(result, output_file)
                    
                    # Add metadata for summary
                    result['k'] = k
                    result['n'] = n
                    result['c'] = c
                    all_results.append(result)
                    
                    processed += 1
                except Exception as e:
                    print(f"\nError processing {gen_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Process additional specific configurations
    for k, n, c in additional_configs:
        gen_file = gen_concepts_dir / f'concept_all_{k}_{n}_{c}.json'
        
        if not gen_file.exists():
            print(f"\nWarning: Generated file not found: {gen_file}")
            continue
        
        output_file = output_dir / f'evaluation_{k}_{n}_{c}.json'
        
        try:
            result = evaluator.evaluate_concept_file(
                gen_file, 
                vocabularies, 
                vocab_embeddings,
                inter_distances,
                use_auto_threshold,
                args.threshold
            )
            evaluator.save_results(result, output_file)
            
            # Add metadata for summary
            result['k'] = k
            result['n'] = n
            result['c'] = c
            all_results.append(result)
            
            processed += 1
        except Exception as e:
            print(f"\nError processing {gen_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary statistics
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY")
    print("="*80)
    
    if all_results:
        summary_data = []
        for r in all_results:
            row = {
                'k': r['k'],
                'n': r['n'],
                'c': r['c'],
                'num_concepts': r['num_generated_concepts'],
                'threshold_mode': 'auto' if r['use_auto_threshold'] else 'fixed'
            }
            
            for vocab_name in vocabularies.keys():
                if vocab_name in r['vocabulary_results']:
                    stats = r['vocabulary_results'][vocab_name]['statistics']
                    row[f'{vocab_name}_threshold'] = stats['threshold_used']
                    row[f'{vocab_name}_mean_sim'] = stats['mean_max_similarity']
                    row[f'{vocab_name}_median_sim'] = stats['median_max_similarity']
                    row[f'{vocab_name}_above_threshold'] = stats['above_threshold']
                    row[f'{vocab_name}_mapped'] = stats['mapped_concepts']
                    row[f'{vocab_name}_total_mappings'] = stats['total_mappings']
                    row[f'{vocab_name}_avg_mappings_per_concept'] = stats['avg_mappings_per_concept']
                    row[f'{vocab_name}_specificity_score'] = stats['specificity_score']
                    row[f'{vocab_name}_few_mappings'] = stats['concepts_with_few_mappings']
                    row[f'{vocab_name}_many_mappings'] = stats['concepts_with_many_mappings']
            
            summary_data.append(row)
        
        # Create summary DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save summary as CSV
        summary_file = output_dir / 'summary.csv'
        df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
        # Print summary table
        print("\nSummary Statistics:")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        print(df.to_string(index=False))
        
        # Comparative analysis
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        print("\n" + "="*80)
        print("POINT 1: Clinical Relevance")
        print("(Generated concepts should be more similar to clinical than general)")
        print("="*80)
        
        for vocab_name in vocabularies.keys():
            mean_col = f'{vocab_name}_mean_sim'
            if mean_col in df.columns:
                print(f"\n{vocab_name.upper()}:")
                print(f"  Overall mean similarity: {df[mean_col].mean():.4f}")
                print(f"  Overall median similarity: {df[mean_col].median():.4f}")
                print(f"  Std dev: {df[mean_col].std():.4f}")
                print(f"  Range: [{df[mean_col].min():.4f}, {df[mean_col].max():.4f}]")
        
        # Clinical vs General comparison
        if 'derm7pt_mean_sim' in df.columns and 'icd10cm_mean_sim' in df.columns:
            clinical_mean = (df['derm7pt_mean_sim'].mean() + df['dermlike_mean_sim'].mean()) / 2
            general_mean = df['icd10cm_mean_sim'].mean()
            improvement = ((clinical_mean - general_mean) / general_mean) * 100
            
            print(f"\n{'='*60}")
            print(f"Clinical (Derm7pt+Dermlike) avg: {clinical_mean:.4f}")
            print(f"General (ICD-10-CM) avg:         {general_mean:.4f}")
            print(f"Improvement:                    {improvement:.1f}%")
            print(f"{'='*60}")
            if improvement > 30:
                print("PASS: Concepts are clinically relevant!")
            else:
                print("FAIL: Insufficient clinical relevance")
        
        print("\n" + "="*80)
        print("POINT 2: Specificity/Representativeness")
        print("(Concepts should map to SOME not ALL manual concepts)")
        print("="*80)
        
        for vocab_name in ['derm7pt', 'dermlike']:
            if f'{vocab_name}_avg_mappings_per_concept' in df.columns:
                avg_mappings = df[f'{vocab_name}_avg_mappings_per_concept'].mean()
                specificity = df[f'{vocab_name}_specificity_score'].mean()
                few = df[f'{vocab_name}_few_mappings'].mean()
                many = df[f'{vocab_name}_many_mappings'].mean()
                
                print(f"\n{vocab_name.upper()}:")
                print(f"  Avg mappings per concept: {avg_mappings:.2f}")
                print(f"  Specificity score: {specificity:.4f}")
                print(f"  Concepts with few mappings (1-3): {few:.1f}")
                print(f"  Concepts with many mappings (>5): {many:.1f}")
                
                if avg_mappings < 5 and few > many:
                    print("  Good: Concepts are specific!")
                elif avg_mappings > 10:
                    print("  Warning: Concepts may be too general")
        
        print("\n" + "="*80)
        print("POINT 3: Cross-Vocabulary Comparison")
        print("(Generated→Clinical should be better than Clinical→Clinical)")
        print("="*80)
        
        if 'derm7pt_mean_sim' in df.columns:
            gen_to_derm7pt = df['derm7pt_mean_sim'].mean()
            gen_to_dermlike = df['dermlike_mean_sim'].mean()
            gen_to_clinical = (gen_to_derm7pt + gen_to_dermlike) / 2
            
            print(f"\nGenerated → Derm7pt:    {gen_to_derm7pt:.4f}")
            print(f"Generated → Dermlike:   {gen_to_dermlike:.4f}")
            print(f"Generated → Clinical:   {gen_to_clinical:.4f}")
            
            if 'derm7pt_vs_dermlike' in cross_vocab_sims:
                derm7pt_to_dermlike = cross_vocab_sims['derm7pt_vs_dermlike']['mean']
                print(f"\nDerm7pt ↔ Dermlike:     {derm7pt_to_dermlike:.4f}")
                
                print(f"\n{'='*60}")
                if gen_to_clinical > derm7pt_to_dermlike:
                    ratio = gen_to_clinical / derm7pt_to_dermlike
                    print(f"Generated concepts are {ratio:.2f}x more similar to clinical")
                    print(f"vocabularies than they are to each other!")
                    print("EXCELLENT mapping quality!")
                else:
                    print("Note: Clinical vocabularies are more similar to each other")
                print(f"{'='*60}")
    
    print(f"\n{'='*80}")
    print(f"Completed! Processed {processed}/{total_files} file combinations")
    print("="*80)


if __name__ == '__main__':
    main()

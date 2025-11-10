import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm


class ConceptMapperGemini:
    def __init__(self):
        """Initialize Gemini model for concept mapping."""
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.mapping_template = """You are a dermatology expert. Your task is to map generated dermatological concepts to a set of established manual concepts based on semantic similarity and clinical relevance.

Generated Concepts:
{generated_concepts}

Manual Concepts (Reference):
{manual_concepts}

Your task:
1. For EACH generated concept, identify which manual concepts are semantically similar or clinically related
2. A generated concept can map to MULTIPLE manual concepts if they are related
3. Consider synonyms, related terms, and clinical associations
4. Only create mappings when there is clear semantic or clinical relationship

Format your response as a JSON object where:
- Keys are the EXACT generated concept strings
- Values are arrays of EXACT manual concept strings that match

Example format:
{{
  "Generated concept 1": ["Manual concept A", "Manual concept B"],
  "Generated concept 2": ["Manual concept C"],
  "Generated concept 3": []
}}

CRITICAL REQUIREMENTS:
1. Use EXACT concept strings from the provided lists (case-sensitive)
2. Return a complete JSON object with ALL generated concepts as keys
3. Use empty arrays [] for concepts with no matches
4. Do not add any explanations, only return the JSON object
"""
    
    def map_concepts_single_run(self, generated_concepts, manual_concepts, max_retries=3):
        """
        Perform a single mapping run using Gemini.
        
        Args:
            generated_concepts: List of generated concept strings
            manual_concepts: List of manual concept strings
            max_retries: Maximum number of retry attempts
        
        Returns:
            Dictionary mapping generated concepts to list of manual concepts
        """
        gen_list = "\n".join([f"- {concept}" for concept in generated_concepts])
        manual_list = "\n".join([f"- {concept}" for concept in manual_concepts])
        
        prompt = self.mapping_template.format(
            generated_concepts=gen_list,
            manual_concepts=manual_list
        )
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    response_text = response.text.strip()
                    
                    # Clean up markdown code blocks
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    elif response_text.startswith('```'):
                        response_text = response_text[3:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    mapping = json.loads(response_text)
                    
                    # Validate mapping structure
                    if isinstance(mapping, dict):
                        # Ensure all generated concepts are present
                        complete_mapping = {}
                        for gen_concept in generated_concepts:
                            if gen_concept in mapping:
                                # Filter to only include valid manual concepts
                                valid_matches = [
                                    m for m in mapping[gen_concept] 
                                    if m in manual_concepts
                                ]
                                complete_mapping[gen_concept] = valid_matches
                            else:
                                complete_mapping[gen_concept] = []
                        
                        return complete_mapping
                
            except json.JSONDecodeError as e:
                print(f"      JSON decode error (attempt {attempt+1}/{max_retries}): {e}")
            except Exception as e:
                print(f"      Error (attempt {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        # Return empty mapping if all retries failed
        return {concept: [] for concept in generated_concepts}
    
    def map_concepts_with_voting(self, generated_concepts, manual_concepts, num_runs=5, vote_threshold=3):
        """
        Map concepts using multiple runs and majority voting.
        
        Args:
            generated_concepts: List of generated concept strings
            manual_concepts: List of manual concept strings
            num_runs: Number of mapping runs to perform (default: 5)
            vote_threshold: Minimum votes needed to keep a mapping (default: 3)
        
        Returns:
            Dictionary with final mappings and vote counts
        """
        print(f"  Running {num_runs} mapping iterations...")
        
        # Store all mappings from each run
        all_mappings = []
        
        for run in range(num_runs):
            print(f"    Run {run+1}/{num_runs}...", end=' ')
            mapping = self.map_concepts_single_run(generated_concepts, manual_concepts)
            all_mappings.append(mapping)
            print("done")
            
            if run < num_runs - 1:
                time.sleep(1)
        
        print(f"  Applying majority voting (threshold: {vote_threshold}/{num_runs})...")
        
        # Count votes for each (generated_concept, manual_concept) pair
        pair_votes = {}
        for gen_concept in generated_concepts:
            pair_votes[gen_concept] = Counter()
            
            for mapping in all_mappings:
                for manual_concept in mapping.get(gen_concept, []):
                    pair_votes[gen_concept][manual_concept] += 1
        
        # Apply voting threshold
        final_mapping = {}
        vote_details = {}
        
        for gen_concept in generated_concepts:
            matches = []
            details = {}
            
            for manual_concept, votes in pair_votes[gen_concept].items():
                details[manual_concept] = votes
                if votes >= vote_threshold:
                    matches.append(manual_concept)
            
            final_mapping[gen_concept] = matches
            vote_details[gen_concept] = details
        
        return {
            'mapping': final_mapping,
            'vote_details': vote_details,
            'num_runs': num_runs,
            'vote_threshold': vote_threshold
        }
    
    def process_file(self, gen_file, manual_file, output_file, num_runs=5, vote_threshold=3):
        """Process a single generated concept file and map to manual concepts."""
        print(f"\n{'='*80}")
        print(f"Processing: {gen_file.name}")
        print(f"Manual concepts: {manual_file.name}")
        
        # Load generated concepts
        with open(gen_file, 'r') as f:
            gen_data = json.load(f)
        
        # Load manual concepts
        with open(manual_file, 'r') as f:
            manual_data = json.load(f)
        
        # Map all concepts
        print("\nMapping 'all' concepts...")
        all_result = self.map_concepts_with_voting(
            gen_data['all'],
            manual_data['all'],
            num_runs,
            vote_threshold
        )
        
        # Map class-specific concepts
        class_results = {}
        for class_name, gen_concepts in gen_data['class_concepts'].items():
            if class_name in manual_data['class_concepts']:
                print(f"\nMapping '{class_name}' concepts...")
                manual_concepts = manual_data['class_concepts'][class_name]
                class_results[class_name] = self.map_concepts_with_voting(
                    gen_concepts,
                    manual_concepts,
                    num_runs,
                    vote_threshold
                )
        
        # Calculate statistics
        all_mapped = sum(1 for matches in all_result['mapping'].values() if matches)
        all_total_mappings = sum(len(matches) for matches in all_result['mapping'].values())
        
        statistics = {
            'total_generated_concepts': len(gen_data['all']),
            'total_manual_concepts': len(manual_data['all']),
            'mapped_concepts': all_mapped,
            'total_mappings': all_total_mappings,
            'num_runs': num_runs,
            'vote_threshold': vote_threshold
        }
        
        # Prepare result
        result = {
            'generated_file': str(gen_file),
            'manual_file': str(manual_file),
            'num_runs': num_runs,
            'vote_threshold': vote_threshold,
            'all_concepts_mapping': all_result['mapping'],
            'all_concepts_votes': all_result['vote_details'],
            'class_concepts_mapping': {
                class_name: res['mapping'] 
                for class_name, res in class_results.items()
            },
            'class_concepts_votes': {
                class_name: res['vote_details']
                for class_name, res in class_results.items()
            },
            'statistics': statistics
        }
        
        # Save result
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nStatistics:")
        print(f"  Total generated concepts: {statistics['total_generated_concepts']}")
        print(f"  Total manual concepts: {statistics['total_manual_concepts']}")
        print(f"  Mapped concepts (with at least 1 match): {statistics['mapped_concepts']}")
        print(f"  Total mappings: {statistics['total_mappings']}")
        print(f"\nSaved to: {output_file}")
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Map generated concepts to manual concepts using Gemini with majority voting'
    )
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of mapping runs for voting (default: 5)')
    parser.add_argument('--vote_threshold', type=int, default=3,
                       help='Minimum votes to keep a mapping (default: 3)')
    parser.add_argument('--dataset', type=str, default='ham10000',
                       help='Dataset name (default: ham10000)')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    gen_concepts_dir = base_path / 'pipeline' / 'concept_extractor' / 'concepts' / 'gemini' / args.dataset
    manual_concepts_dir = base_path / 'eval' / 'manual_concepts'
    output_dir = base_path / 'eval' / 'concept_mappings' / 'gemini' / args.dataset
    
    # Initialize mapper
    mapper = ConceptMapperGemini()
    
    # Define files to process
    k_values = [3, 5, 10]
    n_values = [4]
    c_values = [50, 100, 200]
    
    manual_files = [
        manual_concepts_dir / f'{args.dataset}_dermlike.json',
        manual_concepts_dir / f'{args.dataset}_derm7pt.json'
    ]
    
    # Check manual files exist
    for manual_file in list(manual_files):
        if not manual_file.exists():
            print(f"Warning: Manual file not found: {manual_file}")
            manual_files.remove(manual_file)
    
    if not manual_files:
        print("Error: No manual concept files found")
        return
    
    # Process each combination
    total_files = len(k_values) * len(n_values) * len(c_values) * len(manual_files)
    print(f"\nProcessing {total_files} file combinations...")
    print(f"Runs per mapping: {args.num_runs}")
    print(f"Vote threshold: {args.vote_threshold}")
    
    processed = 0
    for k in k_values:
        for n in n_values:
            for c in c_values:
                gen_file = gen_concepts_dir / f'concept_all_{k}_{n}_{c}.json'
                
                if not gen_file.exists():
                    print(f"\nWarning: Generated file not found: {gen_file}")
                    continue
                
                for manual_file in manual_files:
                    manual_name = manual_file.stem.replace(f'{args.dataset}_', '')
                    output_file = output_dir / manual_name / f'mapping_{k}_{n}_{c}.json'
                    
                    try:
                        mapper.process_file(
                            gen_file, 
                            manual_file, 
                            output_file,
                            args.num_runs,
                            args.vote_threshold
                        )
                        processed += 1
                    except Exception as e:
                        print(f"\nError processing {gen_file.name} -> {manual_file.name}: {e}")
                        import traceback
                        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Completed! Processed {processed}/{total_files} file combinations")


if __name__ == '__main__':
    main()


import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv


class ConceptExtractorGemini:
    def __init__(self, dataset):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.dataset = dataset
        
        if dataset == 'ham10000':
            self.classes = {
                "mel": "Melanoma",
                "nv": "Melanocytic nevi"
            }
        # elif dataset == 'derm7pt':
        #     self.classes = {
        #         "nevus": "Nevus (benign) - various types of benign moles including blue, clark, combined, congenital, dermal, recurrent, and reed/spitz nevi",
        #         "melanoma": "Melanoma (malignant) - dangerous skin cancer in various stages including in situ and invasive melanoma"
        #     }
        # elif dataset == 'cub':
        #     # CUB-200-2011 dataset: 200 bird species
        #     self.classes = self._load_cub_classes()
        # else:
        #     raise ValueError(f"Unknown dataset: {dataset}")
        
        if dataset in ['ham10000', 'derm7pt']:
            self.concept_extraction_template = """You are a dermatology expert analyzing skin lesion images. 

I have analyzed multiple clusters of similar skin lesion images. Below are the descriptive paragraphs for each cluster:

{paragraphs}

Based on these cluster descriptions, please extract EXACTLY {num_concepts} distinct visual and dermatological concepts that are useful for distinguishing between different types of skin lesions.

Requirements:
1. Extract EXACTLY {num_concepts} concepts (no more, no less)
2. Each concept should be a short phrase (2-6 words)
3. Concepts should be descriptive and clinically relevant
4. Avoid redundancy - each concept should be distinct
5. Use dermatological terminology where appropriate

Format your response as a JSON array of exactly {num_concepts} concept strings:
["concept 1", "concept 2", ..., "concept {num_concepts}"]"""
#         elif dataset == 'cub':
#             self.concept_extraction_template = """You are an ornithology expert analyzing bird images. 

# I have analyzed multiple clusters of similar bird images. Below are the descriptive paragraphs for each cluster:

# {paragraphs}

# Based on these cluster descriptions, please extract EXACTLY {num_concepts} distinct visual and morphological concepts that are useful for distinguishing between different bird species.

# Requirements:
# 1. Extract EXACTLY {num_concepts} concepts (no more, no less)
# 2. Each concept should be a short phrase (2-6 words)
# 3. Concepts should be descriptive and ornithologically relevant (e.g., plumage patterns, bill shape, body size)
# 4. Avoid redundancy - each concept should be distinct
# 5. Use ornithological terminology where appropriate

# Format your response as a JSON array of exactly {num_concepts} concept strings:
# ["concept 1", "concept 2", ..., "concept {num_concepts}"]"""
        
        if dataset in ['ham10000', 'derm7pt']:
            self.class_mapping_template = """You are a dermatology expert. I have extracted {num_concepts} visual and dermatological concepts from skin lesion images.

Concepts:
{concept_list}

Disease Classes:
{class_descriptions}

Your task is to map each concept to the disease classes where it is MOST relevant and diagnostically useful. A concept can be mapped to multiple classes if it's relevant to distinguishing or identifying those conditions.

Consider:
- Which concepts are characteristic or diagnostic of specific diseases
- Which concepts help differentiate between diseases
- Clinical relevance of each concept to each disease class

Format your response as a JSON object with class names as keys and arrays of relevant concepts as values:
{{
    "class_name_1": ["concept_a", "concept_b", ...],
    "class_name_2": ["concept_c", "concept_d", ...],
    ...
}}

CRITICAL REQUIREMENTS:
1. Use exact concept phrases from the provided list
2. Use exact class names: {class_names}
3. EACH class MUST have AT LEAST 1 concept assigned (no empty arrays)
4. A concept can appear in multiple classes if relevant
5. Focus on clinically meaningful associations"""
#         elif dataset == 'cub':
#             self.class_mapping_template = """You are a bird expert. I have extracted {num_concepts} visual and morphological concepts from bird images.

# Concepts:
# {concept_list}

# Bird Species:
# {class_descriptions}

# Your task is to map each concept to the bird species where it is MOST relevant and distinguishing. A concept can be mapped to multiple species if it's a characteristic feature of those birds.

# Consider:
# - Which concepts are characteristic or diagnostic of specific bird species
# - Which concepts help differentiate between similar species
# - Visual relevance of each concept to each bird species

# Format your response as a JSON object with class indices (0-199) as string keys and arrays of relevant concepts as values:
# {{
#     "0": ["concept_a", "concept_b", ...],
#     "1": ["concept_c", "concept_d", ...],
#     ...
# }}

# CRITICAL REQUIREMENTS:
# 1. Use exact concept phrases from the provided list
# 2. Use class indices "0" through "199" as keys (as strings)
# 3. EACH bird species MUST have AT LEAST 1 concept assigned (no empty arrays)
# 4. A concept can appear in multiple classes if relevant
# 5. Focus on visually meaningful associations
# 6. Map concepts to ALL 200 bird species
# 7. If a concept doesn't fit well with a species, assign the most relevant general concepts"""
    
    def load_paragraphs(self, paragraph_file):
        """Load cluster paragraphs from JSON file."""
        with open(paragraph_file, 'r') as f:
            data = json.load(f)
        
        paragraphs = []
        for item in data:
            if 'error' not in item and item.get('paragraph'):
                paragraphs.append(f"Cluster {item['cluster_id']}: {item['paragraph']}")
        
        return paragraphs
    
    # def _load_cub_classes(self):
    #     """Load CUB-200-2011 bird species class names."""
    #     base_path = Path(__file__).parent.parent
    #     classes_file = base_path.parent / 'datasets' / 'CUB_200_2011' / 'classes.txt'
        
    #     classes = {}
    #     with open(classes_file, 'r') as f:
    #         for line in f:
    #             class_id, class_name = line.strip().split(' ', 1)
    #             # Convert to 0-indexed
    #             class_idx = int(class_id) - 1
    #             # Clean up class name (remove number prefix and convert underscores)
    #             clean_name = class_name.split('.', 1)[1].replace('_', ' ')
    #             classes[str(class_idx)] = clean_name
        
    #     return classes
    
    def extract_concepts(self, paragraphs, num_concepts, max_retries=3):
        """Extract concepts from merged paragraphs."""
        merged_paragraphs = "\n\n".join(paragraphs)
        
        prompt = self.concept_extraction_template.format(
            paragraphs=merged_paragraphs,
            num_concepts=num_concepts
        )
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    response_text = response.text.strip()
                    
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    concepts = json.loads(response_text)
                    
                    if isinstance(concepts, list) and len(concepts) == num_concepts:
                        return concepts
                    else:
                        if abs(len(concepts) - num_concepts) <= 2:
                            if len(concepts) > num_concepts:
                                concepts = concepts[:num_concepts]
                            return concepts
                
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        return None
    
    def map_concepts_to_classes(self, concepts, max_retries=5):
        """Map concepts to classes using a single run."""
        concept_list = "\n".join([f"- {concept}" for concept in concepts])
        
        # if self.dataset == 'cub':
        #     class_descriptions = "\n".join([f"- {idx}: {name}" for idx, name in list(self.classes.items())[:50]])  # Show first 50 for context
        #     class_descriptions += "\n... (and 150 more bird species)"
        #     class_names = ", ".join([f'"{idx}"' for idx in self.classes.keys()])
        # else:
        class_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in self.classes.items()])
        class_names = ", ".join([f'"{name}"' for name in self.classes.keys()])
        
        prompt = self.class_mapping_template.format(
            num_concepts=len(concepts),
            concept_list=concept_list,
            class_descriptions=class_descriptions,
            class_names=class_names
        )
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    response_text = response.text.strip()
                    
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    mapping = json.loads(response_text)
                    
                    # Validate and fix mapping
                    for class_name in self.classes.keys():
                        if class_name not in mapping or not mapping[class_name]:
                            if not mapping.get(class_name):
                                mapping[class_name] = [concepts[0]] if concepts else []
                    
                    # Validate final mapping structure
                    if self.dataset == 'cub':
                        valid_keys = [k for k in mapping.keys() if k in self.classes.keys()]
                        if len(valid_keys) >= len(self.classes) * 0.5:  # At least 50% coverage
                            # Fill in missing classes
                            complete_mapping = {k: [concepts[0]] if concepts else [] for k in self.classes.keys()}
                            complete_mapping.update({k: v for k, v in mapping.items() if k in self.classes.keys()})
                            return complete_mapping
                    else:
                        if set(mapping.keys()) == set(self.classes.keys()):
                            return mapping
                
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
            
            if attempt < max_retries - 1:
                time.sleep(2)
        
        return None
    
    def process(self, paragraph_file, num_concepts, output_file, k, n):
        """Full pipeline: extract concepts and map to classes."""
        paragraphs = self.load_paragraphs(paragraph_file)
        
        print(f"Extracting {num_concepts} concepts...")
        concepts = self.extract_concepts(paragraphs, num_concepts)
        
        if not concepts:
            print(f"  ERROR: Failed to extract concepts")
            return None
        
        print(f"   Extracted {len(concepts)} concepts")
        
        print(f"Mapping concepts to classes...")
        class_mapping = self.map_concepts_to_classes(concepts)
        
        if not class_mapping:
            print(f"  ERROR: Failed to map concepts to classes")
            return None
        
        print(f"   Mapped concepts to {len(class_mapping)} classes")
        
        result = {
            "all": concepts,
            "class_concepts": class_mapping
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Extract and map concepts to classes using Gemini')
    parser.add_argument('--dataset', type=str, required=True, choices=['ham10000', 'derm7pt', 'cub'],
                        help='Dataset name')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--n', type=int, required=True, help='Number of representative images per cluster')
    parser.add_argument('--c', type=int, required=True, help='Number of concepts to extract')
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent
    paragraph_file = base_path / 'clusters' / 'gemini' / args.dataset / f'paragraph_{args.k}_{args.n}.json'
    output_file = base_path / 'concepts' / 'gemini' / args.dataset / f'concept_all_{args.k}_{args.n}_{args.c}.json'
    
    if not paragraph_file.exists():
        print(f"Error: Paragraph file not found: {paragraph_file}")
        return
    
    extractor = ConceptExtractorGemini(args.dataset)
    result = extractor.process(paragraph_file, args.c, output_file, args.k, args.n)
    
    if result:
        print(f" Completed: {output_file}")


if __name__ == '__main__':
    main()

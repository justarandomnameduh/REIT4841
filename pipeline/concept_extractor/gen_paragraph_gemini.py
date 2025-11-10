import os
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv


class ParagraphGeneratorGemini:
    def __init__(self, dataset='ham10000'):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.dataset = dataset
        
        if dataset in ['ham10000', 'derm7pt']:
            self.prompt_template = """You are analyzing dermatological images from the same cluster. Your task is to identify COMMON VISUAL FEATURES and PATTERNS across these images.

Please analyze these {num_images} representative images from Cluster {cluster_id} and provide a descriptive paragraph focusing on:

- Visible dermatological characteristics (color, texture, patterns, borders)
- Morphological features (shape, symmetry, structure)
- Common visual elements shared across the images
- Distinctive features that make this cluster recognizable

Format your response as a single cohesive paragraph describing the shared visual characteristics of these skin lesion images."""
        elif dataset == 'cub':
            self.prompt_template = """You are analyzing bird images from the same cluster. Your task is to identify COMMON VISUAL FEATURES and PATTERNS across these bird images.

Please analyze these {num_images} representative images from Cluster {cluster_id} and provide a descriptive paragraph focusing on:

- Physical characteristics (colors, patterns, markings on feathers, head, body, wings, tail)
- Body shape and size features (beak shape, wing shape, tail length, body proportions)
- Distinctive visual elements (crests, eye-rings, breast patterns, wing bars)
- Common visual traits shared across these bird images

Format your response as a single cohesive paragraph describing the shared visual characteristics of these bird images."""
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def load_images(self, image_paths):
        """Load images from paths."""
        images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                images.append(image)
        return images
    
    def generate_paragraph(self, cluster_id, image_paths, max_retries=3):
        """Generate descriptive paragraph for a cluster."""
        images = self.load_images(image_paths)
        
        if not images:
            return {
                'cluster_id': cluster_id,
                'paragraph': None,
                'error': f"No images found for cluster {cluster_id}"
            }
        
        prompt = self.prompt_template.format(
            num_images=len(images),
            cluster_id=cluster_id
        )
        
        for attempt in range(max_retries):
            try:
                content = [prompt] + images
                response = self.model.generate_content(content)
                
                if response.text:
                    return {
                        'cluster_id': cluster_id,
                        'paragraph': response.text.strip()
                    }
                else:
                    raise ValueError("Empty response from Gemini API")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        'cluster_id': cluster_id,
                        'paragraph': None,
                        'error': str(e)
                    }
    
    def process_all_clusters(self, representative_file, output_file, delay=2.0):
        """Process all clusters and generate paragraphs."""
        with open(representative_file, 'r') as f:
            representatives = json.load(f)
        
        results = []
        
        for i, cluster_data in enumerate(representatives):
            cluster_id = cluster_data['cluster_id']
            image_paths = cluster_data['representative_images']
            
            print(f"Processing cluster {cluster_id}/{len(representatives)-1}...")
            result = self.generate_paragraph(cluster_id, image_paths)
            results.append(result)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            if i < len(representatives) - 1:
                time.sleep(delay)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate descriptive paragraphs for clusters using Gemini')
    parser.add_argument('--dataset', type=str, required=True, choices=['ham10000', 'derm7pt', 'cub'],
                        help='Dataset name')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--n', type=int, required=True, help='Number of representative images per cluster')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between API calls (seconds)')
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent
    representative_file = base_path / 'clusters' / args.dataset / f'representative_{args.k}_{args.n}.json'
    output_dir = base_path / 'clusters' / 'gemini' / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'paragraph_{args.k}_{args.n}.json'
    
    if not representative_file.exists():
        print(f"Error: Representative file not found: {representative_file}")
        return
    
    generator = ParagraphGeneratorGemini(args.dataset)
    generator.process_all_clusters(representative_file, output_file, args.delay)
    print(f"Completed: {output_file}")


if __name__ == '__main__':
    main()

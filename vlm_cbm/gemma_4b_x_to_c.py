import os
import torch
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import pipeline
from datasets import Dataset

# Enable TensorFloat32 precision for better performance
torch.set_float32_matmul_precision('high')

# --- Configuration ---
# Paths for the dataset
BASE_DATA_PATH = "."
IMAGE_DIR = os.path.join(BASE_DATA_PATH, "ISIC2018_Task3_Validation_Input")
OUTPUT_JSON_PATH = "skin_lesion_concepts.json"  # Output JSON file name

# Model configuration
MODEL_NAME = "google/gemma-3-4b-it"  # Use the image-text-to-text model

# Define an enhanced prompt for generating descriptive concepts
SYSTEM_PROMPT = """You are a specialized dermatology image analyzer. Your task is to provide detailed, visual descriptions of skin lesions without making any diagnostic conclusions.

Focus on the following visual characteristics:
1. Overall shape: Is it symmetric or asymmetric? Round, oval, irregular?
2. Border characteristics: Are the borders smooth, well-defined or irregular, fuzzy, notched?
3. Color: What colors are present (brown, black, tan, red, blue, white)? Is there color variation or uniformity?
4. Structure: Does it have a raised appearance, flat, or both? Are there any visible patterns, dots, streaks, or networks?
5. Size: Does it appear small, medium, or large relative to the surrounding skin?
6. Texture: Does it look smooth, rough, scaly, crusty, or ulcerated?

Provide only factual, visually observable descriptions. List 5-10 specific visual characteristics in bullet point format without any diagnostic interpretation."""

def preprocess_image(image_path):
    """Load an image from disk and prepare it for the model"""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def create_messages(image):
    """Create the formatted messages for Gemma 3 model"""
    if image is None:
        return None
        
    return [
        {
            "role": "system", 
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the visual characteristics of this skin lesion in detail, focusing on the aspects mentioned above. Provide only factual descriptions without any diagnostic conclusions."}
            ]
        }
    ]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    print(f"Loading model: {MODEL_NAME}...")
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_NAME,
            device=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model with pipeline API: {str(e)}")
        return

    # Check if image directory exists
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Image directory not found at {os.path.abspath(IMAGE_DIR)}")
        return

    # Get list of images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"No JPG images found in {IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} images. Preparing dataset...")
    
    # Prepare dataset entries
    dataset_entries = []
    for image_file in tqdm(image_files, desc="Preparing images"):
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        # Load the image
        image = preprocess_image(image_path)
        
        # Create messages
        messages = create_messages(image)
        
        # Add to dataset entries
        dataset_entries.append({
            "image_id": image_id,
            "image_path": image_path,
            "image": image,
            "messages": messages
        })
    
    # Create a HuggingFace dataset
    dataset = Dataset.from_list(dataset_entries)
    
    # Define processing function for the pipeline
    def generate_description(example):
        if example["messages"] is None:
            return {"description": "Error generating description"}
            
        try:
            output = pipe(
                text=example["messages"],
                max_new_tokens=300,
                do_sample=True
            )
            description = output[0]["generated_text"][-1]["content"]
            return {"description": description.strip()}
        except Exception as e:
            print(f"Error processing {example['image_id']}: {str(e)}")
            return {"description": "Error generating description"}
    
    # Process the dataset in batches
    print("Generating descriptions...")
    results = dataset.map(
        generate_description,
        batched=False,  # Process one at a time (Gemma pipeline doesn't support batching for this use case)
        desc="Generating descriptions"
    )
    
    # Collect results
    all_concepts = []
    for result in results:
        concept_entry = {
            "image_id": result["image_id"],
            "description": result["description"]
        }
        all_concepts.append(concept_entry)
        print(f"Image: {result['image_id']} - Generated description")
    
    # Save all concepts to a JSON file
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_concepts, f, indent=2)
    
    print(f"Concepts saved to {OUTPUT_JSON_PATH}")
    
    # Clean up
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("ISIC 2018 Task 3 Concept Generation using Google Gemma 3")
    print("--------------------------------------------------------------------")
    print(f"Dataset path: {os.path.abspath(BASE_DATA_PATH)}")
    main()
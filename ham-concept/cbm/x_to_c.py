import os
import torch
import json
import pandas as pd  # Added for CSV reading
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import pipeline

# Enable TensorFloat32 precision for better performance
torch.set_float32_matmul_precision('high')

# --- Configuration ---
# Paths for the dataset
# Adjusted BASE_DATA_PATH to be relative to the script location or an absolute path
# Assuming the script is in ham-concept/cbm/ and datasets are in ham-concept/ham_concept_dataset/
BASE_PROJECT_PATH = "/home/nqmtien/THESIS/REIT4841/ham-concept"  # Goes up to ham-concept/
HAM_CONCEPT_DATASET_PATH = f"{BASE_PROJECT_PATH}/ham_concept_dataset"

IMAGE_DIR = f"{HAM_CONCEPT_DATASET_PATH}/ISIC2018_Task3_Training_Input"  # Updated image directory
METADATA_CSV_PATH = f"{HAM_CONCEPT_DATASET_PATH}/Datasets/metadata/val.csv"  # Path to val.csv
OUTPUT_JSON_PATH = "skin_lesion_concept_abbreviations.json"  # Output JSON file name

# Number of rows to process from val.csv
NUM_ROWS_TO_PROCESS = 100

# Model configuration
MODEL_NAME = "google/gemma-3-12b-it"  # Use the image-text-to-text model

# Define an enhanced prompt for generating descriptive concepts
SYSTEM_PROMPT = """You are an expert dermatoscopic image analyzer. Your task is to identify and list the presence of specific dermatoscopic features in the provided skin lesion image.
Refer to the following list of features, their abbreviations, and detailed explanations. For each feature you identify in the image, state its abbreviation. If a feature is not present, do not mention it.

FEATURE DEFINITIONS WITH ABBREVIATIONS:

ESA: Eccentrically located structureless area (any colour except skin colour, white and grey) <definition> Relevant for this criterion is the noncentral location of the structureless area.  White and grey structureless areas are covered by the melanoma criteria "White lines or white structureless area" and "grey patterns", as well as by the nevus criterion "Melanoma simulator", regardless of their location.  </definition>

GP: Grey patterns <definition> This criterion refers to grey structureless areas (regardless of their localization) as well as to grey circles, lines, dots or globules.  There is overlap between this criterion and the nevus criterion "Melanoma simulator".  </definition>

PV: Polymorphous vessels <definition> This signifies several types of vessels occurring together, for example, dot-like and curved vessels.  </definition>

BDG: Black dots or globules in the periphery of the lesion <definition> In particular, the dots or globules are not symmetrically arranged with other patterns (for example, reticular lines).  </definition>

WLSA: White lines or white structureless area <definition> The localization of the structures is irrelevant for this criterion.  There is overlap with the nevus criterion "melanoma simulator".  </definition>

SPC: Symmetrical combination of patterns and/or colours <definition> Two types of symmetric combinations are possible: 1. A pattern and/or colour is evenly distributed within another pattern and/or colour (e.g., dark brown dots on light brown reticular lines).  2. A pattern and/or colour is centrally located within another pattern and/or colour (dark brown reticular lines within light brown reticular lines, dark brown/black globules within dark brown/black radial lines, light brown structureless area centrally within dark brown reticular lines).  Melanoma simulators may also show symmetrical combinations, e.g., grey-white central structureless area within skin-coloured globules in the case of a nonpigmented Spitz nevus).  </definition>

PRL: Pseudopods or radial lines at the lesional margin involving the entire lesional circumference <definition> Strictly speaking, this criterion is a special case of a symmetrical pattern combination.  It is listed separately here to distinguish it from the melanoma criterion "Pseudopods or radial lines at the lesional margin that do not involve the entire lesional circumference".  </definition>

PLF: Parallel lines in the furrows (acral lesions only) <definition> In acral nevi, pigment may also be located on the ridges; in this case, parallel lines in the furrows are found mainly at the lesional margin.  </definition>

APC: Asymmetric combination of multiple patterns and/or colours in the absence of other melanoma criteria <definition> Despite the asymmetry, no other clear-cut melanoma criteria are present; thus, the lesion can in principle be evaluated as benign.  </definition>



Based on the image provided, list the abbreviations of all the features you observe, separated by commas. The abbreviations should be in uppercase letters. If you identify multiple features, separate them with commas. If no features are present, respond with "None". Do not include any additional text or explanations in your response.

Example output format:
PRL, APC, GP
"""

# User prompt asking for concept abbreviations
USER_PROMPT_TEXT = "Based on the image provided, list the abbreviations of all the features you observe, separated by commas, according to the feature definitions provided in the system prompt. If no features are present, respond with \"None\"."

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

    # Load metadata CSV
    try:
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
        print(f"Successfully loaded metadata from {METADATA_CSV_PATH}")
    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found at {METADATA_CSV_PATH}")
        return
    except Exception as e:
        print(f"Error reading metadata CSV: {str(e)}")
        return

    # Select the first N rows
    image_ids_to_process = metadata_df['image_id'].head(NUM_ROWS_TO_PROCESS).tolist()
    if not image_ids_to_process:
        print(f"No image IDs found in the first {NUM_ROWS_TO_PROCESS} rows of {METADATA_CSV_PATH}")
        return

    print(f"Found {len(image_ids_to_process)} images to process from CSV. Starting concept generation...")
    
    # Process images
    all_outputs = []
    
    for image_id in tqdm(image_ids_to_process, desc="Processing images"):
        image_file_name = f"{image_id}.jpg"  # Assuming images are JPEGs
        image_path = os.path.join(IMAGE_DIR, image_file_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            all_outputs.append({
                "image_id": image_id,
                "concepts": ["Error: Image file not found"]
            })
            continue
            
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')
                
            # Setup the message format for Gemma 3
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": USER_PROMPT_TEXT}
                    ]
                }
            ]
            
            raw_output = pipe(
                messages,
                images=[image],  # Pass the image as a keyword argument
                max_new_tokens=100,
                do_sample=False
            )
            
            if raw_output and isinstance(raw_output, list) and raw_output[0].get("generated_text"):
                generated_content = raw_output[0]["generated_text"]
                if isinstance(generated_content, list) and generated_content:
                    concept_str = generated_content[-1]["content"]
                elif isinstance(generated_content, str):
                    concept_str = generated_content
                else:
                    concept_str = "Error: Could not parse model output"
                    print(f"Warning: Unexpected model output structure for {image_id}: {raw_output}")
            else:
                concept_str = "Error: Empty or invalid model output"
                print(f"Warning: Empty or invalid model output for {image_id}: {raw_output}")

            concept_str = concept_str.strip()
            
            if concept_str.upper() == "NONE" or not concept_str:
                concepts_list = []
            elif "Error:" in concept_str:
                concepts_list = [concept_str]
            else:
                concepts_list = [c.strip().upper() for c in concept_str.split(',') if c.strip()]
            
            output_entry = {
                "image_id": image_id,
                "concepts": concepts_list
            }
            
            all_outputs.append(output_entry)
                
        except Exception as e:
            print(f"Error processing {image_id} ({image_file_name}): {str(e)}")
            all_outputs.append({
                "image_id": image_id,
                "concepts": [f"Error processing image: {str(e)}"]
            })
    
    # Save all concepts to a JSON file
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_outputs, f, indent=2)
    
    print(f"\nConcept abbreviations saved to {OUTPUT_JSON_PATH}")
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("ISIC 2018 Task 3 Concept Abbreviation Generation using Google Gemma 3")
    print("--------------------------------------------------------------------")
    print(f"Reading images from: {os.path.abspath(IMAGE_DIR)}")
    print(f"Reading metadata from: {os.path.abspath(METADATA_CSV_PATH)}")
    print(f"Processing first {NUM_ROWS_TO_PROCESS} images.")
    main()
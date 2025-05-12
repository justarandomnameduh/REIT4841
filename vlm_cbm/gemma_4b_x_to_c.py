import os
import torch
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import pipeline

# Enable TensorFloat32 precision for better performance
torch.set_float32_matmul_precision('high')

# --- Configuration ---
# Paths for the dataset
BASE_DATA_PATH = "../ham-concept/ham_concept_dataset"
IMAGE_DIR = os.path.join(BASE_DATA_PATH, "ISIC2018_Task3_Validation_Input")
OUTPUT_JSON_PATH = "skin_lesion_concepts.json"  # Output JSON file name

# Model configuration
MODEL_NAME = "google/gemma-3-4b-it"  # Use the image-text-to-text model

# Define an enhanced prompt for generating descriptive concepts
SYSTEM_PROMPT = """You are an expert dermatoscopic image analyzer. Your task is to identify and list the presence of specific dermatoscopic features in the provided skin lesion image.
Refer to the following list of features, their abbreviations, and detailed explanations. For each feature you identify in the image, state its abbreviation. If a feature is not present, do not mention it.

FEATURE DEFINITIONS:
TRBL: Thick reticular or branched lines <definition> Thick lines refer to lines that are at least as wide as the areas between them.  </definition>

ESA: Eccentrically located structureless area (any colour except skin colour, white and grey) <definition> Relevant for this criterion is the noncentral location of the structureless area.  White and grey structureless areas are covered by the melanoma criteria "White lines or white structureless area" and "grey patterns", as well as by the nevus criterion "Melanoma simulator", regardless of their location.  </definition>

GP: Grey patterns <definition> This criterion refers to grey structureless areas (regardless of their localization) as well as to grey circles, lines, dots or globules.  There is overlap between this criterion and the nevus criterion "Melanoma simulator".  </definition>

PV: Polymorphous vessels <definition> This signifies several types of vessels occurring together, for example, dot-like and curved vessels.  </definition>

BDG: Black dots or globules in the periphery of the lesion <definition> In particular, the dots or globules are not symmetrically arranged with other patterns (for example, reticular lines).  </definition>

WLSA: White lines or white structureless area <definition> The localization of the structures is irrelevant for this criterion.  There is overlap with the nevus criterion "melanoma simulator".  </definition>

PLR: Parallel lines on ridges (acral lesions only) <definition> What is relevant here is that the furrows (often only visible at the lesional margin) are not pigmented.  </definition>

PES: Pigmentation extends beyond the area of the scar (only after excision) <definition> This refers to cases in which a recurrent melanoma grows beyond the margin of the excised area.  </definition>

PIF: Pigmentation invades the openings of hair follicles (facial lesions) <definition> Pigmentation of advanced melanomas on the face may involve the openings of hair follicles.  </definition>

OPC: Only one pattern and only one colour <definition> For example, only brown reticular lines or only skin-coloured globules.  This would also include a blue nevus that presents only as blue and structureless.  </definition>

SPC: Symmetrical combination of patterns and/or colours <definition> Two types of symmetric combinations are possible: 1. A pattern and/or colour is evenly distributed within another pattern and/or colour (e.g., dark brown dots on light brown reticular lines).  2. A pattern and/or colour is centrally located within another pattern and/or colour (dark brown reticular lines within light brown reticular lines, dark brown/black globules within dark brown/black radial lines, light brown structureless area centrally within dark brown reticular lines).  Melanoma simulators may also show symmetrical combinations, e.g., grey-white central structureless area within skin-coloured globules in the case of a nonpigmented Spitz nevus).  </definition>

PRL: Pseudopods or radial lines at the lesional margin involving the entire lesional circumference <definition> Strictly speaking, this criterion is a special case of a symmetrical pattern combination.  It is listed separately here to distinguish it from the melanoma criterion "Pseudopods or radial lines at the lesional margin that do not involve the entire lesional circumference".  </definition>

PLF: Parallel lines in the furrows (acral lesions only) <definition> In acral nevi, pigment may also be located on the ridges; in this case, parallel lines in the furrows are found mainly at the lesional margin.  </definition>

PDES: Pigmentation does not extend beyond the area of the scar (only after excision) <definition> Recurrent nevi typically do not extend beyond the scar area.  </definition>

APC: Asymmetric combination of multiple patterns and/or colours in the absence of other melanoma criteria <definition> Despite the asymmetry, no other clear-cut melanoma criteria are present; thus, the lesion can in principle be evaluated as benign.  </definition>

MS: Melanoma simulator <definition> This criterion is used to distinguish melanomas from nevus types that can (and in the present cases do) exhibit melanoma features.  Relevant features are, e.g., white lines or grey areas in blue, Reed and Spitz nevi.  The underlying idea is that the lesion in this particular case is in principle considered benign despite the appearance of melanoma criteria.  </definition>

PRLC: Pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference <definition> What is relevant here is that the pseudopods/radial lines affect only part of the lesion.  The colour of the pattern is irrelevant in this case.  </definition>

Based on the image provided, list the abbreviations of all the features you observe, separated by commas.
Example output format:
TRBL, ESA, GP
"""

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

    print(f"Found {len(image_files)} images. Starting concept generation...")
    
    # Process images
    all_concepts = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGE_DIR, image_file)
        
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
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe the visual characteristics of this skin lesion in detail, focusing on the aspects mentioned above. Provide only factual descriptions without any diagnostic conclusions."}
                    ]
                }
            ]
            
            output = pipe(
                text=messages,
                max_new_tokens=300,
                do_sample=True
            )
            description = output[0]["generated_text"][-1]["content"]
            
            # Create a JSON entry for this image
            concept_entry = {
                "image_id": image_id,
                "description": description.strip()
            }
            
            all_concepts.append(concept_entry)
            print(f"Image: {image_id} - Generated description")
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            # Add an entry even if there's an error, just with an error message
            all_concepts.append({
                "image_id": image_id,
                "description": "Error generating description"
            })
    
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
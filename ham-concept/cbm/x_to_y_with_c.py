import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import pipeline

# Enable TensorFloat32 precision for better performance
torch.set_float32_matmul_precision('high')

# --- Configuration ---
BASE_PROJECT_PATH = "/home/nqmtien/THESIS/REIT4841/ham-concept"
HAM_CONCEPT_DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "ham_concept_dataset")

IMAGE_DIR = os.path.join(HAM_CONCEPT_DATASET_PATH, "ISIC2018_Task3_Training_Input")
METADATA_CSV_PATH = os.path.join(HAM_CONCEPT_DATASET_PATH, "Datasets/metadata/val.csv")
OUTPUT_JSON_FILENAME = "predictions_mel_nv.json"  # Output JSON file name

# Number of rows to process from val.csv (-1 to process all)
NUM_ROWS_TO_PROCESS = -1 

# Model configuration
MODEL_NAME = "google/gemma-3-4b-it"

# Define the target categories
TARGET_CATEGORIES = ["MEL", "NV"]
FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus"
}

# Define the prompt template for MEL/NV classification
SYSTEM_PROMPT = f"""You are a specialized skin lesion classifier. Your task is to classify dermoscopic images of skin lesions as either Melanoma (MEL) or Melanocytic nevus (NV).
First, provide the classification (MEL or NV).
Then, on a new line, provide reasoning that led to your classification, prefixed with "REASONS: ". The reasoning should be as detailed as possible, cover all the aspect that you deem relevant. The reasoning should be presented as a single paragraph, not as a list or bullet points. There is no limit to the number of reasons you can provide, but they should be relevant to the classification task.
"""

USER_PROMPT_TEXT = "Classify this skin lesion image as either MEL (Melanoma) or NV (Melanocytic nevus). Provide the category abbreviation (MEL or NV) and reasoning for your classification as per the system instructions."

def parse_model_response(response_text):
    """Parse the model's text response to extract prediction and reasons."""
    prediction = "NV"  # Default prediction
    reasons = "General appearance" # Default reasons as a single string

    if not response_text:
        print("Empty response. Defaulting to NV and general reasons.")
        return {"prediction": prediction, "reason": reasons}

    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    
    parsed_prediction = False
    parsed_reasons = False

    if lines:
        # Check first line for prediction
        first_line_upper = lines[0].upper()
        if "MEL" == first_line_upper:
            prediction = "MEL"
            parsed_prediction = True
        elif "NV" == first_line_upper:
            prediction = "NV"
            parsed_prediction = True
        elif "MELANOMA" in first_line_upper and "NEVUS" not in first_line_upper:
             prediction = "MEL"
             parsed_prediction = True
        elif "NEVUS" in first_line_upper and "MELANOMA" not in first_line_upper:
             prediction = "NV"
             parsed_prediction = True

        # Check for REASONS line
        reason_line_index = -1
        for i, line in enumerate(lines):
            if line.upper().startswith("REASONS:"):
                reason_line_index = i
                break
        
        if reason_line_index != -1:
            reasons_str = lines[reason_line_index][len("REASONS:"):].strip()
            if reasons_str:
                reasons = reasons_str # Store the entire paragraph as a single string
                parsed_reasons = True
            else:
                print(f"Reasons string was empty after 'REASONS:': '{lines[reason_line_index]}'. Using default reasons.")
        
        if not parsed_prediction:
            # Fallback: check keywords in the whole response if not parsed from first line
            response_upper = response_text.upper()
            is_mel = "MELANOMA" in response_upper or "MEL" in response_upper
            is_nv = "NEVUS" in response_upper or "NV" in response_upper

            if is_mel and not is_nv:
                prediction = "MEL"
            elif is_nv and not is_mel:
                prediction = "NV"
            elif is_mel and is_nv:
                print(f"Ambiguous keywords for MEL/NV in response: '{response_text}'. Defaulting prediction to NV.")
            else:
                print(f"Could not map response to MEL or NV from full text: '{response_text}'. Defaulting prediction to NV.")
        
        if not parsed_reasons:
             print(f"No 'REASONS:' line found or reasons were empty in response: '{response_text}'. Using default reasons.")
    else:
        print(f"Response text was empty or only whitespace: '{response_text}'. Defaulting prediction and reasons.")

    return {"prediction": prediction, "reason": reasons}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_json_path = os.path.join(script_dir, OUTPUT_JSON_FILENAME)

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    hf_pipe = None
    try:
        hf_pipe = pipeline(
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

    # Load metadata CSV to get image IDs
    try:
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
        print(f"Successfully loaded metadata from {METADATA_CSV_PATH} with {len(metadata_df)} entries.")
    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found at {METADATA_CSV_PATH}")
        return
    except Exception as e:
        print(f"Error reading metadata CSV: {str(e)}")
        return

    # Select image IDs to process
    if NUM_ROWS_TO_PROCESS != -1:
        image_ids_to_process = metadata_df['image_id'].head(NUM_ROWS_TO_PROCESS).tolist()
    else:
        image_ids_to_process = metadata_df['image_id'].tolist()
    
    if not image_ids_to_process:
        print(f"No image IDs found to process from {METADATA_CSV_PATH}")
        return

    print(f"Found {len(image_ids_to_process)} images to process from CSV. Starting predictions...")
    
    all_outputs = []
    
    for image_id in tqdm(image_ids_to_process, desc="Processing images"):
        image_file_name = f"{image_id}.jpg"
        image_path = os.path.join(IMAGE_DIR, image_file_name)
        
        current_prediction = "NV" # Default prediction
        current_reasons = "General appearance" # Default reasons

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Recording error for this image.")
            current_prediction = "Error: Image file not found"
            current_reasons = "Image file not found at path" # Changed to string
            all_outputs.append({"image_id": image_id, "prediction": current_prediction, "reason": current_reasons})
            continue
            
        try:
            pil_image = Image.open(image_path).convert('RGB')
                
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": USER_PROMPT_TEXT}
                    ]
                }
            ]
            
            raw_model_output = hf_pipe(
                text=messages,
                max_new_tokens=512, # Increased max_new_tokens
                do_sample=False 
            )
            
            response_text = "NV" # Default if extraction fails
            if raw_model_output and isinstance(raw_model_output, list) and raw_model_output[0].get("generated_text"):
                generated_parts = raw_model_output[0]["generated_text"]
                if generated_parts and isinstance(generated_parts, list) and generated_parts[-1].get("content"):
                    response_text = generated_parts[-1]["content"]
                else:
                    print(f"Warning: Unexpected generated_text structure for {image_id}. Full output: {raw_model_output}")
            else:
                print(f"Warning: Unexpected or empty model output for {image_id}. Full output: {raw_model_output}")

            print(f"Image: {image_id}, Raw Response: '{response_text}'")
            parsed_output = parse_model_response(response_text)
            current_prediction = parsed_output["prediction"]
            current_reasons = parsed_output["reason"]
            
        except Exception as e:
            print(f"Error processing {image_id} ({image_file_name}): {str(e)}. Recording error.")
            current_prediction = f"Error: {str(e)}"
            current_reasons = f"Error during processing: {str(e)}" # Changed to string

        all_outputs.append({"image_id": image_id, "prediction": current_prediction, "reason": current_reasons})
    
    # Save all predictions to a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(all_outputs, f, indent=2)
    
    print(f"\nPredictions saved to {output_json_path}")
    
    # Clean up
    if hf_pipe is not None and hasattr(hf_pipe, 'model'):
        del hf_pipe.model
    if hf_pipe is not None:
        del hf_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("ISIC 2018 Task 3 - MEL/NV Classification using Google Gemma 3")
    print("--------------------------------------------------------------------")
    print(f"Reading images from: {os.path.abspath(IMAGE_DIR)}")
    print(f"Reading metadata from: {os.path.abspath(METADATA_CSV_PATH)}")
    if NUM_ROWS_TO_PROCESS == -1:
        print("Processing all images found in metadata.")
    else:
        print(f"Processing first {NUM_ROWS_TO_PROCESS} images from metadata.")
    main()

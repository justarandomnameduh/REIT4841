import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, pipeline

# --- Configuration ---
# Paths for the dataset
BASE_DATA_PATH = "."
IMAGE_DIR = os.path.join(BASE_DATA_PATH, "ISIC2018_Task3_Validation_Input")
GROUND_TRUTH_PATH = os.path.join(BASE_DATA_PATH, "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")
OUTPUT_CSV_PATH = "predictions_gemma.csv"  # Output CSV file name

# Model configuration
MODEL_NAME = "google/gemma-3-4b-it"  # Use the image-text-to-text model

# Define the categories based on the task
CATEGORIES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma", 
    "AKIEC": "Actinic keratosis / Bowen's disease",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion"
}

# Define the prompt template for classification
SYSTEM_PROMPT = """You are a specialized skin lesion classifier. Your task is to classify dermoscopic images of skin lesions into one of the following categories:
MEL: Melanoma
NV: Melanocytic nevus
BCC: Basal cell carcinoma
AKIEC: Actinic keratosis / Bowen's disease
BKL: Benign keratosis
DF: Dermatofibroma
VASC: Vascular lesion

Respond with only the abbreviation from the above list that best represents the lesion in the image."""

def map_response_to_category(response):
    """Map the model's text response to one of the predefined categories"""
    if not response:
        print(f"Empty response. Defaulting to NV.")
        return "NV"
        
    response = response.strip().upper()
    
    for cat in CATEGORIES:
        if cat == response:
            return cat
            
    for cat in CATEGORIES:
        if cat in response:
            return cat
        
    response_lower = response.lower()
    for cat, full_name in FULL_NAMES.items():
        if full_name.lower() in response_lower:
            return cat
    
    # Default NV if no match (based on dataset distribution)
    print(f"Could not map response to category: '{response}'. Defaulting to NV.")
    return "NV"

def evaluate_accuracy(predictions_df, ground_truth_df):
    """Calculate accuracy metrics for the predictions"""
    pred_df = predictions_df.copy()
    gt_df = ground_truth_df.copy()
    
    # Ensure we're working with the same set of images
    merged_df = pred_df.merge(gt_df, on='image')
    
    correct = 0
    total = len(merged_df)
    
    for idx, row in merged_df.iterrows():
        # Find predicted label (from our predictions)
        pred_label = None
        for category in CATEGORIES:
            if row[category + '_x'] == 1.0:  # _x suffix for predictions
                pred_label = category
                break
        
        # Find true label (from ground truth)
        true_label = None
        for category in CATEGORIES:
            if row[category + '_y'] == 1.0:  # _y suffix for ground truth
                true_label = category
                break
        
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, merged_df

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

    # Load ground truth for later evaluation
    ground_truth_path = Path(GROUND_TRUTH_PATH)
    if ground_truth_path.exists():
        ground_truth_df = pd.read_csv(ground_truth_path)
        print(f"Loaded ground truth data with {len(ground_truth_df)} entries.")
    else:
        print(f"Warning: Ground truth file not found at {ground_truth_path}")
        ground_truth_df = None

    # Get list of images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"No JPG images found in {IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} images. Starting predictions...")
    
    # Process images
    all_predictions = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        # Initialize prediction row with all classes set to 0.0
        prediction = {"image": image_id}
        for category in CATEGORIES:
            prediction[category] = 0.0
            
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
                        {"type": "text", "text": "Classify this skin lesion image into one of the categories listed above. Respond with only the category abbreviation."}
                    ]
                }
            ]
            
            output = pipe(
                text=messages,
                max_new_tokens=20,
                do_sample=True
            )
            response = output[0]["generated_text"][-1]["content"]
            
            print(f"Image: {image_id}, Response: '{response}'")
            
            # Map to a valid category
            predicted_category = map_response_to_category(response)
            prediction[predicted_category] = 1.0
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            prediction["NV"] = 1.0
            
        all_predictions.append(prediction)
    
    # Create and save predictions dataframe
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.1f')
    print(f"Predictions saved to {OUTPUT_CSV_PATH}")
    
    # Clean up
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    
    # Evaluate results if ground truth is available
    if ground_truth_df is not None:
        accuracy, correct, total, results_df = evaluate_accuracy(predictions_df, ground_truth_df)
        print(f"Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Show a sample of predictions vs ground truth
        print("\nSample predictions (first 5):")
        
        # Safely display the first few results
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            # Find predicted and actual labels
            pred_class = next((c for c in CATEGORIES if row[f'{c}_x'] == 1.0), 'None')
            true_class = next((c for c in CATEGORIES if row[f'{c}_y'] == 1.0), 'None')
            
            print(f"Image: {row['image']}, Predicted: {pred_class}, Actual: {true_class}")
            
        # Display class-wise performance
        print("\nClass-wise performance:")
        class_correct = {c: 0 for c in CATEGORIES}
        class_total = {c: 0 for c in CATEGORIES}
        
        for idx, row in results_df.iterrows():
            # Get true label
            true_class = next((c for c in CATEGORIES if row[f'{c}_y'] == 1.0), None)
            if true_class:
                # Get predicted label
                pred_class = next((c for c in CATEGORIES if row[f'{c}_x'] == 1.0), None)
                class_total[true_class] += 1
                if pred_class == true_class:
                    class_correct[true_class] += 1
        
        # Print class-wise accuracy
        for category in CATEGORIES:
            if class_total[category] > 0:
                accuracy = class_correct[category] / class_total[category]
                print(f"  {category} ({FULL_NAMES[category]}): {accuracy:.4f} ({class_correct[category]}/{class_total[category]})")
            else:
                print(f"  {category} ({FULL_NAMES[category]}): No samples")

if __name__ == "__main__":
    print("ISIC 2018 Task 3 Classification using Google Gemma 3")
    print("--------------------------------------------------------------------")
    print(f"Dataset path: {os.path.abspath(BASE_DATA_PATH)}")
    main()
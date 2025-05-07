import os
import torch
import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Enable TensorFloat32 precision for better performance
torch.set_float32_matmul_precision('high')

# --- Configuration ---
# Paths for the dataset
BASE_DATA_PATH = "isic_2018_task3/data"
IMAGE_DIR = os.path.join(BASE_DATA_PATH, "validation_input/ISIC2018_Task3_Validation_Input")
GROUND_TRUTH_PATH = os.path.join(BASE_DATA_PATH, "validation_ground_truth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")
CONCEPTS_PATH = "skin_lesion_concepts.json"  # Path to the concepts file generated earlier
OUTPUT_CSV_PATH = "predictions_qwen.csv"  # Output CSV file name

# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"

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

def extract_category(response):
    """Extract a category from the model's response using a more robust approach"""
    # Clean up the response
    response = response.strip()
    
    # Look for final answer after thinking process
    # Common patterns in the response
    answer_patterns = [
        r'final answer:?\s*([A-Z]+)',  # "Final answer: MEL" or similar
        r'the diagnosis is:?\s*([A-Z]+)',  # "The diagnosis is: BCC" or similar
        r'classification:?\s*([A-Z]+)',  # "Classification: VASC" or similar
        r'I classify this as:?\s*([A-Z]+)',  # "I classify this as: NV" or similar
        r'my answer is:?\s*([A-Z]+)',  # "My answer is: DF" or similar
        r'the lesion is:?\s*([A-Z]+)',  # "The lesion is: AKIEC" or similar
        r'category:?\s*([A-Z]+)',  # "Category: BKL" or similar
        r'\b(MEL|NV|BCC|AKIEC|BKL|DF|VASC)\b',  # Just the category on its own
    ]
    
    # Try all patterns to find a match
    for pattern in answer_patterns:
        matches = re.search(pattern, response, re.IGNORECASE)
        if matches:
            potential_category = matches.group(1).upper()
            if potential_category in CATEGORIES:
                return potential_category
    
    # Check for full category names
    response_lower = response.lower()
    for cat, full_name in FULL_NAMES.items():
        if full_name.lower() in response_lower:
            return cat
    
    # Default to NV if no match (based on dataset distribution)
    print(f"Could not extract category from response: '{response[:100]}...'")  # Showing first 100 chars
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

    # Load model and use the pipeline API
    print(f"Loading model: {MODEL_NAME}...")
    try:
        # Using the pipeline API as recommended for Qwen3
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load the concepts file
    try:
        with open(CONCEPTS_PATH, 'r') as f:
            concepts = json.load(f)
        print(f"Loaded concepts data with {len(concepts)} entries.")
        
        # Create a dictionary for quick lookup by image_id
        concepts_dict = {concept["image_id"]: concept["description"] for concept in concepts}
    except Exception as e:
        print(f"Error loading concepts file: {str(e)}")
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
        
        # Initialize prediction row with all classes set to 0.0
        prediction = {"image": image_id}
        for category in CATEGORIES:
            prediction[category] = 0.0
            
        try:
            # Get the description for this image
            if image_id in concepts_dict:
                description = concepts_dict[image_id]
            else:
                print(f"Warning: No description found for {image_id}. Skipping.")
                prediction["NV"] = 1.0  # Default to NV if no description
                all_predictions.append(prediction)
                continue
            
            # Create the proper message format for Qwen3
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a specialized dermatological diagnosis assistant. "
                        "Your task is to classify skin lesions based on visual descriptions "
                        "into exactly one of seven categories.\n\n"
                        "CATEGORIES:\n"
                        "- MEL: Melanoma - A type of skin cancer that can be irregular in shape, with varied colors "
                        "(often including shades of black, brown, red, or blue), asymmetrical appearance, and uneven borders.\n"
                        "- NV: Melanocytic nevus - A common mole, typically symmetrical with regular borders, "
                        "uniform color (usually brown, tan, or skin-colored), and smaller size.\n"
                        "- BCC: Basal cell carcinoma - Often appears as a pearly, waxy bump, or a flat, "
                        "flesh-colored or brown scar-like lesion. May have visible blood vessels and can have "
                        "a central depression or ulceration.\n"
                        "- AKIEC: Actinic keratosis / Bowen's disease - Rough, scaly patches that can be red, "
                        "pink, or brown. May appear as crusty, scaly areas with irregular borders.\n"
                        "- BKL: Benign keratosis - Waxy, stuck-on appearance, can be brown, black, or tan with "
                        "well-defined borders, often with a 'warty' surface texture.\n"
                        "- DF: Dermatofibroma - Firm, raised growths that are usually round, reddish-brown to pink, "
                        "and may dimple when pinched.\n"
                        "- VASC: Vascular lesion - Bright red or purple in color, can be raised or flat, well-defined, "
                        "and may blanch under pressure. Includes cherry angiomas and pyogenic granulomas.\n\n"
                        "You can use <think> tags to think through your reasoning process, then present your final answer. "
                        "Your final answer should be just the category abbreviation on its own line preceded by 'FINAL ANSWER:'"
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Based on the following description of a skin lesion, classify it into exactly one of "
                        f"the seven categories listed above.\n\n"
                        f"Description: {description}\n\n"
                        f"You can think through your reasoning using <think> tags, then provide your final answer "
                        f"in the format 'FINAL ANSWER: XXX' where XXX is just the category abbreviation (e.g., MEL, NV, BCC, etc.)."
                    )
                }
            ]
            
            # Generate response from Qwen using the pipeline
            output = pipe(
                messages,
                max_new_tokens=32768,  # Increased to allow thinking
                temperature=0.1,
                top_k=2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=151643,  # Qwen's EOS token
                eos_token_id=151643,
                return_full_text=False
            )
            
            # Extract the response
            response = output[0]["generated_text"]
            print(f"Image: {image_id}, Response: '{response}...'")  # Show first 50 chars
            
            # Extract the category using our improved function
            predicted_category = extract_category(response)
            prediction[predicted_category] = 1.0
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            prediction["NV"] = 1.0  # Default to NV if there's an error
            
        all_predictions.append(prediction)
    
    # Create and save predictions dataframe
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.1f')
    print(f"Predictions saved to {OUTPUT_CSV_PATH}")
    
    # Clean up
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
    print("ISIC 2018 Task 3 Classification using Qwen 3-8B")
    print("--------------------------------------------------------------------")
    print(f"Dataset path: {os.path.abspath(BASE_DATA_PATH)}")
    main()
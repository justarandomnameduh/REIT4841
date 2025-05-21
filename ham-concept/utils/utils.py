import os
import pandas as pd

def get_true_label(image_id, csv_file):
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file not found at {csv_file}")
    df = pd.read_csv(csv_file)
    
    if 'image_id' not in df.columns or 'dx' not in df.columns:
        raise ValueError("CSV file must contain 'image_id' and 'dx' columns")
    
    matching_rows = df[df['image_id'] == image_id]
    if matching_rows.empty:
        return None
    
    label = matching_rows.iloc[0]['dx'].upper() 
    return label


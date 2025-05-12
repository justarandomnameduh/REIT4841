import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
INPUT_CSV_FILE = '../ham_concept_dataset/Datasets/metadata/metadata_ground_truth.csv'
OUTPUT_TRAIN_FILE = '../ham_concept_dataset/Datasets/metadata/train.csv'
OUTPUT_VAL_FILE = '../ham_concept_dataset/Datasets/metadata/val.csv'
OUTPUT_TEST_FILE = '../ham_concept_dataset/Datasets/metadata/test.csv'

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# For reproducibility of the split
RANDOM_STATE = 42

# Optional: If you want to stratify the split based on a particular column
# (e.g., a class label to ensure proportional representation in each set).
# Set to None if no stratification is needed.
# Example: STRATIFY_COLUMN = 'label'
STRATIFY_COLUMN = None
# --- End Configuration ---

def split_data(input_file, train_file, val_file, test_file,
               train_ratio, val_ratio, test_ratio,
               random_state, stratify_col=None):
    """
    Splits a CSV file into train, validation, and test sets.

    Args:
        input_file (str): Path to the input CSV file.
        train_file (str): Path to save the training data CSV.
        val_file (str): Path to save the validation data CSV.
        test_file (str): Path to save the test data CSV.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        random_state (int): Seed for random number generator for reproducibility.
        stratify_col (str, optional): Column name to use for stratified splitting.
                                      Defaults to None.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file '{input_file}': {e}")
        return

    if df.empty:
        print(f"Error: Input file '{input_file}' is empty.")
        return

    print(f"Original dataset shape: {df.shape}")

    # Ensure ratios sum to 1 (approximately)
    if not (0.999 < (train_ratio + val_ratio + test_ratio) < 1.001):
        print("Error: Ratios for train, validation, and test must sum to 1.0.")
        print(f"Current sum: {train_ratio + val_ratio + test_ratio}")
        return

    # Determine stratification array
    stratify_array = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    if stratify_col and stratify_col not in df.columns:
        print(f"Warning: Stratification column '{stratify_col}' not found in the CSV. Proceeding without stratification.")
        stratify_array = None
    elif stratify_col:
        print(f"Stratifying based on column: '{stratify_col}'")


    # First split: separate out the training data
    # The remaining data will be (val_ratio + test_ratio) of the original
    train_df, remaining_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio), # Size of the 'remaining' set
        random_state=random_state,
        stratify=stratify_array
    )

    # Second split: split the remaining data into validation and test
    # The test set size here is relative to the 'remaining_df'
    # test_ratio_of_remaining = test_ratio / (val_ratio + test_ratio)
    if (val_ratio + test_ratio) == 0: # Avoid division by zero if only train split is desired
        val_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)
    else:
        test_ratio_of_remaining = test_ratio / (val_ratio + test_ratio)
        stratify_array_remaining = remaining_df[stratify_col] if stratify_col and stratify_col in remaining_df.columns else None

        val_df, test_df = train_test_split(
            remaining_df,
            test_size=test_ratio_of_remaining,
            random_state=random_state,
            stratify=stratify_array_remaining
        )

    # Save the splits to CSV files
    try:
        train_df.to_csv(train_file, index=False)
        print(f"\nTraining data saved to '{train_file}', shape: {train_df.shape} ({len(train_df)/len(df):.2%})")

        val_df.to_csv(val_file, index=False)
        print(f"Validation data saved to '{val_file}', shape: {val_df.shape} ({len(val_df)/len(df):.2%})")

        test_df.to_csv(test_file, index=False)
        print(f"Test data saved to '{test_file}', shape: {test_df.shape} ({len(test_df)/len(df):.2%})")

        print("\nSplitting complete.")
        print(f"Total rows: {len(df)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Val rows:   {len(val_df)}")
        print(f"Test rows:  {len(test_df)}")
        print(f"Sum of split rows: {len(train_df) + len(val_df) + len(test_df)}")

    except Exception as e:
        print(f"Error writing output CSV files: {e}")


if __name__ == "__main__":
    # --- Create a dummy CSV for testing if it doesn't exist ---
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"'{INPUT_CSV_FILE}' not found. Creating a dummy CSV for demonstration.")
        num_rows = 1000
        data = {
            'id': range(num_rows),
            'feature1': [i * 0.5 for i in range(num_rows)],
            'feature2': [f'value_{i%10}' for i in range(num_rows)],
            # Example 'label' column for stratification testing
            'label': ['A'] * int(num_rows * 0.7) + ['B'] * int(num_rows * 0.2) + ['C'] * int(num_rows * 0.1)
        }
        # Shuffle labels a bit if num_rows is small to ensure all labels appear
        import random
        random.shuffle(data['label'])
        if len(data['label']) < num_rows: # Adjust if rounding caused issues
            data['label'].extend(['A'] * (num_rows - len(data['label'])))

        dummy_df = pd.DataFrame(data)
        dummy_df.to_csv(INPUT_CSV_FILE, index=False)
        print(f"Dummy '{INPUT_CSV_FILE}' created with {num_rows} rows.")
        # If you want to test stratification with the dummy data, uncomment the next line:
        # STRATIFY_COLUMN = 'label'
    # --- End dummy CSV creation ---

    split_data(
        INPUT_CSV_FILE,
        OUTPUT_TRAIN_FILE,
        OUTPUT_VAL_FILE,
        OUTPUT_TEST_FILE,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_STATE,
        STRATIFY_COLUMN
    )
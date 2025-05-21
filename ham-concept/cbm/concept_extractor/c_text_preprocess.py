import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
import yake  # For keyword extraction
import os
import sys

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.utils import get_true_label

# --- NLTK Resource Downloads ---
# Ensure these are downloaded. The script will check and prompt if missing.
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet') # Commented out
# nltk.download('omw-1.4') # Commented out
# nltk.download('averaged_perceptron_tagger')

# --- Configuration ---
INPUT_JSON_FILENAME = "img-reason_paragraphs-raw.json"
OUTPUT_JSON_FILENAME = "processed_key_phrases.json"
CSV_FILE_PATH = "/home/nqmtien/THESIS/REIT4841/ham-concept/ham_concept_dataset/Datasets/metadata/val.csv"

def preprocess_text(text_input):  # Renamed 'text' to 'text_input' for clarity
    """
    Cleans, normalizes, tokenizes, extracts noun phrases, keywords,
    removes stop words, and lemmatizes text.
    Returns a dictionary with processed components.
    """
    if text_input is None:
        return {"processed_tokens": [], "key_phrases": []}

    current_text = ""
    if isinstance(text_input, str):
        current_text = text_input
    elif isinstance(text_input, list):
        # Join list elements. Ensure elements are strings.
        string_elements = [str(s) for s in text_input if isinstance(s, str)]
        current_text = " ".join(string_elements)
    else:
        # Unsupported type for processing
        print(f"Warning: preprocess_text received an unsupported type: {type(text_input)}. Returning empty results.")
        return {"lemmatized_tokens": [], "noun_phrases": [], "keywords": []}

    # 1. Initial Cleaning: Remove "REASONS:" prefix (case-insensitive) and strip whitespace
    cleaned_text = re.sub(r"^[Rr][Ee][Aa][Ss][Oo][Nn][Ss]:\s*", "", current_text).strip()

    # If text becomes empty after stripping "REASONS:" and whitespace
    if not cleaned_text:
        return {"processed_tokens": [], "key_phrases": []}

    # 2. Normalization for general use (lowercase)
    normalized_text_for_keywords_np = cleaned_text.lower()

    # 3. Keyword/Keyphrase Extraction (using YAKE! on minimally processed text)
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=10, features=None)
    keywords_with_scores = kw_extractor.extract_keywords(cleaned_text)  # Use text with minimal changes
    keywords = [kw for kw, score in keywords_with_scores]

    # 4. Noun Phrase Extraction
    text_for_pos = re.sub(r'[^\w\s.\'-]', ' ', normalized_text_for_keywords_np)
    text_for_pos = re.sub(r'\s+', ' ', text_for_pos).strip()
    tokens_for_pos = word_tokenize(text_for_pos)
    tagged_tokens = pos_tag(tokens_for_pos)

    grammar = r"""
        NP: {<DT>?<JJ.*>*<NN.*>+}          # Optional Determiner, Adjective(s), Noun(s)
            {<CD><JJ.*>*<NN.*>+}         # Cardinal number, Adjective(s), Noun(s)
            {<NN.*>+<IN><DT>?<JJ.*>*<NN.*>+} # Noun(s) + Prep + Opt. Det + Adj(s) + Noun(s) (e.g. "areas of pigmentation")
            {<RB.*>*<JJ.*>*<NN.*>+}      # Adverb(s), Adjective(s), Noun(s)
            {<NNP.*>+}                   # Proper Noun(s)
    """
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(tagged_tokens)

    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        np_text = " ".join(word for word, tag in subtree.leaves())
        np_text = re.sub(r'^[.\'-]|[.\'-]$', '', np_text).strip()
        if np_text:
            noun_phrases.append(np_text)

    # 5. Further Normalization for tokenization and lemmatization (remove all non-alphanumeric)
    text_for_lemmatization = re.sub(r'[^\w\s]', ' ', normalized_text_for_keywords_np)
    text_for_lemmatization = re.sub(r'\s+', ' ', text_for_lemmatization).strip()

    # 6. Tokenization for lemmatization
    tokens_for_lem = word_tokenize(text_for_lemmatization)

    # 7. Stop Word Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens_for_lem if word not in stop_words and len(word) > 1]

    # 8. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Combine noun phrases and keywords to get key_phrases
    key_phrases = sorted(list(set(noun_phrases + keywords)))
    
    return {
        "processed_tokens": lemmatized_tokens,
        "key_phrases": key_phrases
    }

def extract_image_id_from_data(entry, index):
    """
    Try to extract image_id from the JSON entry or generate a placeholder if not found.
    """
    # First, try to get an image_id directly from the entry
    if isinstance(entry, dict) and "image_id" in entry:
        return entry["image_id"]
    
    # For img-reason_paragraphs-raw.json format, there's no image_id field
    # We'll use a placeholder based on the index
    return f"UNKNOWN_IMAGE_{index+1}"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_json_path = os.path.join(script_dir, INPUT_JSON_FILENAME)
    output_json_path = os.path.join(script_dir, OUTPUT_JSON_FILENAME)

    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    processed_data = []
    for i, entry in enumerate(data):
        # Handle entries that might come from predictions_mel_nv.json (with image_id)
        # or from img-reason_paragraphs-raw.json (without image_id)
        image_id = extract_image_id_from_data(entry, i)
        
        # Get the reason text - in img-reason_paragraphs-raw.json it's just the "reason" field
        original_reason = entry.get("reason", "")
        
        print(f"Processing entry {i+1}/{len(data)}: {image_id}")
        
        # Get true label from the CSV file if image_id is available
        label = None
        if "UNKNOWN_IMAGE" not in image_id:
            try:
                label = get_true_label(image_id, CSV_FILE_PATH)
                if label:
                    print(f"  Found true label for {image_id}: {label}")
                else:
                    print(f"  Could not find true label for {image_id}")
            except Exception as e:
                print(f"  Error getting true label: {str(e)}")
        
        # Process the text to extract meaningful concepts
        processed_result = preprocess_text(original_reason)
        
        # Log info about the nature of the reason text
        if original_reason is None:
            print(f"  Info: 'reason' for entry #{i+1} was null in JSON. Processed to empty concepts.")
        elif isinstance(original_reason, str) and not original_reason.strip():
            print(f"  Info: 'reason' for entry #{i+1} was an empty string. Processed to empty concepts.")
        
        # Ensure 'original_reason' in the output JSON is not None
        current_output_reason = original_reason
        if original_reason is None:
            current_output_reason = ""

        processed_data.append({
            "image_id": image_id,
            "label": label,
            "original_text": current_output_reason,
            "processed_tokens": processed_result["processed_tokens"],
            "key_phrases": processed_result["key_phrases"]
        })

    # Count how many entries have labels
    entries_with_labels = sum(1 for entry in processed_data if entry.get("label"))
    
    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"\nProcessed concepts saved to {output_json_path}")
    print(f"Total entries processed: {len(processed_data)}")
    print(f"Entries with true labels: {entries_with_labels}")
    
    if processed_data:
        print(f"\nExample of the first processed entry:")
        print(json.dumps(processed_data[0], indent=2))

if __name__ == "__main__":
    resources_to_check = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    all_resources_available = True
    print("Checking and downloading NLTK resources if necessary...")
    for resource_name, resource_path in resources_to_check.items():
        try:
            nltk.data.find(resource_path)
            print(f" - Resource '{resource_name}' already available.")
        except LookupError:
            print(f" - Resource '{resource_name}' not found. Attempting download...")
            try:
                nltk.download(resource_name, quiet=False)
                nltk.data.find(resource_path)
                print(f"   Successfully downloaded and verified '{resource_name}'.")
            except Exception as e:
                print(f"   Failed to download or verify '{resource_name}'. Error: {e}")
                print(f"   Please try running: nltk.download('{resource_name}') manually in a Python console.")
                all_resources_available = False
        except Exception as e:
            print(f"   An unexpected error occurred while checking for resource '{resource_name}': {e}")
            all_resources_available = False

    if not all_resources_available:
        print("\nOne or more NLTK resources could not be automatically acquired or verified. Please check the messages above and try manual downloads if suggested.")
        exit()
    print("All required NLTK resources are available or have been successfully downloaded.\n")

    main()

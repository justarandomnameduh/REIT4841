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

# --- NLTK Resource Downloads ---
# Ensure these are downloaded. The script will check and prompt if missing.
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet') # Commented out
# nltk.download('omw-1.4') # Commented out
# nltk.download('averaged_perceptron_tagger')

# --- Configuration ---
INPUT_JSON_FILENAME = "predictions_mel_nv.json"
OUTPUT_JSON_FILENAME = "processed_concepts.json"

def preprocess_text(text_input):  # Renamed 'text' to 'text_input' for clarity
    """
    Cleans, normalizes, tokenizes, extracts noun phrases, keywords,
    removes stop words, and lemmatizes text.
    Returns a dictionary with processed components.
    """
    if text_input is None:
        return {"lemmatized_tokens": [], "noun_phrases": [], "keywords": []}

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
        return {"lemmatized_tokens": [], "noun_phrases": [], "keywords": []}

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

    return {
        "lemmatized_tokens": lemmatized_tokens,
        "noun_phrases": sorted(list(set(noun_phrases))),
        "keywords": keywords
    }

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
        image_id = entry.get("image_id")
        prediction = entry.get("prediction")
        original_reason = entry.get("reason", "") # Retains current behavior: default to "" if key missing

        print(f"Processing entry {i+1}/{len(data)}: {image_id}")

        # Always call preprocess_text.
        # It's designed to handle str, list, or None (if original_reason was explicitly null in JSON).
        processed_result = preprocess_text(original_reason)

        # Update logging based on the nature of original_reason
        if original_reason is None: # This case occurs if JSON had "reason": null
            print(f"Info: 'reason' for image_id {image_id} was null in JSON. Processed to empty concepts.")
        elif isinstance(original_reason, str) and not original_reason.strip(): # Covers "" or "   "
            print(f"Info: 'reason' for image_id {image_id} was an empty string. Processed to empty concepts.")
        elif isinstance(original_reason, list) and \
             not any(isinstance(s, str) and s.strip() for s in original_reason): # Covers [] or list of empty strings
            print(f"Info: 'reason' for image_id {image_id} was an empty list or list of empty strings. Processed to empty concepts.")
        # preprocess_text itself will warn if original_reason is an unexpected type (e.g. int) and return empty.

        # Ensure 'original_reason' in the output JSON is not None.
        current_output_reason = original_reason
        if original_reason is None:
            current_output_reason = ""

        processed_data.append({
            "image_id": image_id,
            "prediction": prediction,
            "original_reason": current_output_reason,
            "processed_reason_tokens": processed_result["lemmatized_tokens"],
            "noun_phrases": processed_result["noun_phrases"],
            "keywords": processed_result["keywords"]
        })

    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"\nProcessed concepts saved to {output_json_path}")
    print(f"Total entries processed: {len(processed_data)}")
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

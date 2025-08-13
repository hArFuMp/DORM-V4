import os
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# Path to the dataset folder
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# List of Parquet files
TRAIN_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if 'train' in f and f.endswith('.parquet')]
VALID_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if 'test' in f and f.endswith('.parquet')]

# Tokenizer file path (relative to main.py)
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'tokenizer.json')

def process_and_save(files, output_filename):
    """
    Reads Parquet files, tokenizes data, and saves results to a .bin file.
    """
    print(f"Starting {output_filename} generation...")
    all_tokens = []
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

    for file_path in tqdm(files, desc=f"Processing {output_filename}"):
        df = pd.read_parquet(file_path)
        # Assumes a 'text' column. May need adjustment for actual data.
        if 'text' in df.columns:
            text_data = df['text'].tolist()
            # Concatenate all text into a single large string
            full_text = '\n'.join(text_data)
            # Tokenize
            tokens = tokenizer.encode(full_text)
            all_tokens.extend(tokens)

    # Convert to NumPy array
    arr = np.array(all_tokens, dtype=np.uint16) # uint16 is sufficient for GPT-2 vocab size < 65535

    # Save to .bin file
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, 'wb') as f:
        f.write(arr.tobytes())
    
    print(f"{output_filename} generation complete. Total {len(arr)} tokens saved.")

if __name__ == '__main__':
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer file({TOKENIZER_PATH}) not found.")
        print("Please ensure tokenizer.json is in the project root.")
    else:
        process_and_save(TRAIN_FILES, 'train.bin')
        process_and_save(VALID_FILES, 'val.bin')
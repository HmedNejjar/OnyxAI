import numpy as np
from pathlib import Path
from BPE import BPE
import time

"""
Dataset tokenization script

This script handles the tokenization of large text datasets using the BPE tokenizer
and saves the tokenized data as binary numpy files for efficient loading during training.

Key features:
- Loads existing tokenizer or trains a new one if needed
- Tokenizes large text files efficiently
- Saves tokenized data as compact numpy arrays
- Provides detailed timing and compression statistics
"""

# Configuration parameters
VOCAB_SIZE = 22000  # Size of the tokenizer vocabulary
PARENT_FOLDER = Path(r'G:\Projects\Python\OnyxAI')

# Paths to files
TOKENIZER_PATH = PARENT_FOLDER / Path(r'Tokenizer/merges.json')  # Saved tokenizer
TRAIN_TEXT_PATH = PARENT_FOLDER / Path(r'Corpus/TinyStoriesV2-train.txt')  # Training text
VAL_TEXT_PATH = PARENT_FOLDER / Path(r'Corpus/TinyStoriesV2-valid.txt')  # Validation text
TRAIN_TOKENS_PATH = PARENT_FOLDER / Path(r'Corpus/train_tokens.npy')  # Output tokenized training data
VAL_TOKENS_PATH = PARENT_FOLDER / Path(r'Corpus/val_tokens.npy')  # Output tokenized validation data

def tokenize_and_save(text_path: Path, tokens_path: Path, tokenizer: BPE) -> None:
    """
    Tokenize a text file and save the tokenized data as a binary numpy file.
    
    Args:
        text_path (Path): Path to the input text file
        tokens_path (Path): Path to save the tokenized data
        tokenizer (BPE): BPE tokenizer instance to use for tokenization
    """
    print(f"\n  Processing: {text_path.name}")
    print(f"    Loading text...")
    
    # Measure time to load text
    start_time = time.time()
    
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    load_time = time.time() - start_time
    print(f"    Loaded {len(text):,} characters in {load_time:.2f}s")
    
    # Measure time to tokenize
    print(f"    Tokenizing...")
    tokenize_start = time.time()
    tokens = tokenizer.encode(text)
    tokenize_time = time.time() - tokenize_start
    
    print(f"    Generated {len(tokens):,} tokens in {tokenize_time:.2f}s")
    print(f"    Tokenization speed: {len(text)/tokenize_time:,.0f} chars/sec")
    
    # Measure time to save
    print(f"    Saving to {tokens_path.name}...")
    save_start = time.time()
    np.save(tokens_path, np.array(tokens, dtype=np.uint32))  # Use uint32 for memory efficiency
    save_time = time.time() - save_start
    
    # Calculate statistics
    file_size = tokens_path.stat().st_size
    compression = len(text) / len(tokens)
    
    print(f"    Saved {file_size:,} bytes in {save_time:.2f}s")
    print(f"    Compression: {len(text):,} chars → {len(tokens):,} tokens ({compression:.2f}x)")

if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Tokenization")
    print("=" * 60)
    
    # Create directories if they don't exist
    print(f"\n[1] Checking paths...")
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"    Tokenizer path: {TOKENIZER_PATH}")
    print(f"    Train text:     {TRAIN_TEXT_PATH}")
    print(f"    Validation text: {VAL_TEXT_PATH}")
    
    # Check if tokenizer exists
    if TOKENIZER_PATH.exists():
        print(f"\n[2] Loading existing tokenizer from {TOKENIZER_PATH}...")
        tokenizer = BPE(VOCAB_SIZE)
        tokenizer.load(TOKENIZER_PATH)
        print(f"    ✓ Tokenizer loaded successfully")
    else:
        print(f"\n[2] Training new tokenizer (vocab_size={VOCAB_SIZE:,})...")
        print(f"    Training on: {TRAIN_TEXT_PATH.name}")
        
        if not TRAIN_TEXT_PATH.exists():
            print(f"\n✗ Error: Training file not found at {TRAIN_TEXT_PATH}")
            exit(1)
        
        with open(TRAIN_TEXT_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"    Loaded {len(text):,} characters for training")
        print(f"    Training tokenizer... (this may take a while)")
        
        start_time = time.time()
        tokenizer = BPE(VOCAB_SIZE)
        tokenizer.train(text, save_path=TOKENIZER_PATH)
        training_time = time.time() - start_time
        
        print(f"    ✓ Training completed in {training_time:.2f} seconds")
    
    # Tokenize datasets
    print(f"\n[3] Tokenizing datasets...")
    print(f"    {'─' * 50}")
    
    if TRAIN_TEXT_PATH.exists():
        tokenize_and_save(TRAIN_TEXT_PATH, TRAIN_TOKENS_PATH, tokenizer)
    else:
        print(f"  ✗ Training file not found: {TRAIN_TEXT_PATH}")
    
    if VAL_TEXT_PATH.exists():
        tokenize_and_save(VAL_TEXT_PATH, VAL_TOKENS_PATH, tokenizer)
    else:
        print(f"  ✗ Validation file not found: {VAL_TEXT_PATH}")
    
    # Summary
    print(f"\n[4] Summary")
    print(f"    {'─' * 50}")
    print(f"    Tokenizer:        {TOKENIZER_PATH.name}")
    print(f"    Train tokens:     {TRAIN_TOKENS_PATH.name if TRAIN_TOKENS_PATH.exists() else 'Not created'}")
    print(f"    Validation tokens: {VAL_TOKENS_PATH.name if VAL_TOKENS_PATH.exists() else 'Not created'}")
    
    print("\n" + "=" * 60)
    print("Tokenization complete!")
    print("=" * 60)
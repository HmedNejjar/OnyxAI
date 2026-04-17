from BPE import BPE
from pathlib import Path
import time

"""
BPE Tokenizer Training Script

This script trains a Byte-Pair Encoding (BPE) tokenizer on a specified corpus
and saves the trained tokenizer for later use. It provides detailed statistics
about the training process and verifies the tokenizer with a test example.
"""

# Configuration
FILE_PATH = Path(r'G:\Projects\Python\OnyxAI\Corpus\bookcorpus_p1.txt')  # Path to training corpus
SAVE_TOKENIZER = Path(r'bpe_30k_vocab.json')  # Path to save trained tokenizer
VOCAB_SIZE = 30000  # Target vocabulary size

if __name__ == "__main__":
    print("=" * 60)
    print("BPE Tokenizer Training")
    print("=" * 60)
    
    # Check if corpus exists
    if not FILE_PATH.exists():
        print(f"\n✗ Error: Corpus file not found at {FILE_PATH.absolute()}")
        print("  Please ensure the corpus file exists before training.")
        exit(1)
    
    # Load corpus
    print(f"\n[1] Loading corpus...")
    print(f"    File: {FILE_PATH.absolute()}")
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    file_size = FILE_PATH.stat().st_size
    print(f"    Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"    Characters: {len(text):,}")
    
    # Initialize tokenizer
    print(f"\n[2] Initializing BPE tokenizer...")
    print(f"    Target vocabulary size: {VOCAB_SIZE:,} tokens")
    
    encoder = BPE(vocab_size=VOCAB_SIZE)
    
    # Train tokenizer
    print(f"\n[3] Training tokenizer...")
    print(f"    This may take a few minutes for large corpora...")
    
    start_time = time.time()
    ids = encoder.train(text, SAVE_TOKENIZER)
    training_time = time.time() - start_time
    
    # Calculate statistics
    original_bytes = len(text.encode('utf-8'))
    compression_ratio = original_bytes / len(ids)
    
    # Display results
    print(f"\n[4] Training Complete!")
    print(f"    {'─' * 50}")
    print(f"    Training time      : {training_time:.2f} seconds")
    print(f"    Original bytes     : {original_bytes:,}")
    print(f"    Tokens generated   : {len(ids):,}")
    print(f"    Compression ratio  : {compression_ratio:.2f}x")
    print(f"    Bytes per token    : {original_bytes / len(ids):.2f}")
    print(f"    {'─' * 50}")
    
    # Save location
    print(f"\n[5] Output Files:")
    print(f"    Tokenizer saved to: {SAVE_TOKENIZER.absolute()}")
    
    # Quick test
    print(f"\n[6] Quick Verification:")
    test_text = "Hello, world!"
    test_ids = encoder.encode(test_text)
    test_decoded = encoder.decode(test_ids)
    
    print(f"    Test text  : \"{test_text}\"")
    print(f"    Tokenized  : {test_ids}")
    print(f"    Decoded    : \"{test_decoded}\"")
    print(f"    Roundtrip  : {'✓ Success' if test_text == test_decoded else '✗ Failed'}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
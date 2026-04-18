from BPE import BPE
from pathlib import Path
import time

"""
BPE Tokenizer Training Script 

Train a Byte-Pair Encoding (BPE) tokenizer on large corpora
by loading them in memory-efficient chunks. It provides detailed statistics
about the training process and compression achieved.

The chunked approach:
1. Splits corpus into ~10MB chunks (customizable)
2. Splits chunks intelligently at word boundaries (whitespace)
3. Trains tokenizer incrementally on chunks
4. Returns total token count for compression tracking
"""

# Configuration
FILE_PATH = Path(r'G:\Projects\Python\OnyxAI\Corpus\bookcorpus_p1.txt')  # Path to training corpus
SAVE_TOKENIZER = Path(r'bpe_30k_vocab.json')  # Path to save trained tokenizer
VOCAB_SIZE = 30000  # Target vocabulary size
CHUNK_SIZE = 10_000_000  # 10 MB chunks

if __name__ == "__main__":
    print("=" * 70)
    print("BPE Tokenizer Training (Chunked Loading)")
    print("=" * 70)
    
    # Check if corpus exists
    if not FILE_PATH.exists():
        print(f"\n✗ Error: Corpus file not found at {FILE_PATH.absolute()}")
        print("  Please ensure the corpus file exists before training.")
        exit(1)
    
    # Get file info
    file_size = FILE_PATH.stat().st_size
    file_size_mb = file_size / 1024 / 1024
    
    print(f"\n[1] Corpus Information:")
    print(f"    File: {FILE_PATH.absolute()}")
    print(f"    Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"    Chunk size: {CHUNK_SIZE / 1024 / 1024:.1f} MB")
    
    # Initialize tokenizer
    print(f"\n[2] Initializing BPE tokenizer...")
    print(f"    Target vocabulary size: {VOCAB_SIZE:,} tokens")
    
    encoder = BPE(vocab_size=VOCAB_SIZE)
    
    # Train tokenizer on chunks
    print(f"\n[3] Training tokenizer on corpus (chunked loading)...")
    print(f"    This streams through your corpus without loading it all at once")
    
    start_time = time.time()
    total_tokens = encoder.train_chunked(
        file_path=FILE_PATH,
        save_path=SAVE_TOKENIZER,
        chunk_size=CHUNK_SIZE,
        show_progress=True
    )
    training_time = time.time() - start_time
    
    # Calculate statistics
    compression_ratio = file_size / total_tokens
    
    # Display results
    print(f"\n[4] Training Complete!")
    print(f"    {'─' * 60}")
    print(f"    Training time        : {training_time:.2f} seconds")
    print(f"    Original size        : {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"    Total tokens         : {total_tokens:,}")
    print(f"    Compression ratio    : {compression_ratio:.2f}x")
    print(f"    Average bytes/token  : {file_size / total_tokens:.2f}")
    print(f"    {'─' * 60}")
    
    # Estimated space savings
    bytes_per_token = 4  # uint32
    estimated_size = total_tokens * bytes_per_token / 1024 / 1024
    saved_space = file_size_mb - estimated_size
    savings_percent = (saved_space / file_size_mb) * 100
    
    print(f"\n[5] Storage Estimates:")
    print(f"    Original corpus      : {file_size_mb:.2f} MB")
    print(f"    Tokenized (uint32)   : {estimated_size:.2f} MB")
    print(f"    Space saved          : {saved_space:.2f} MB ({savings_percent:.1f}%)")
    
    # Save location
    print(f"\n[6] Output Files:")
    print(f"    Tokenizer saved to   : {SAVE_TOKENIZER.absolute()}")
    
    # Quick verification test
    print(f"\n[7] Quick Verification:")
    test_text = "Hello, world! This is a test of the tokenizer."
    test_ids = encoder.encode(test_text)
    test_decoded = encoder.decode(test_ids)
    
    print(f"    Test text    : \"{test_text}\"")
    print(f"    Tokenized    : {test_ids}")
    print(f"    Decoded      : \"{test_decoded}\"")
    print(f"    Roundtrip OK : {'✓ Yes' if test_text == test_decoded else '✗ No'}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
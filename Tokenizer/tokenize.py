from BPE import BPE
from pathlib import Path
import time

"""
Corpus Tokenization Script (with Chunked Loading)

This script tokenizes a large text corpus and saves the token IDs as a numpy
file for efficient training data loading. Perfect for large datasets that don't
fit in memory.

Process:
1. Load pretrained BPE tokenizer
2. Stream through corpus in chunks
3. Tokenize each chunk
4. Write tokens to memory-mapped numpy file
5. Return total token count for training loop
"""

# Configuration
TOKENIZER_PATH = Path(r'G:\Projects\Python\OnyxAI\Tokenizer\bpe_30k_vocab.json')  # Trained tokenizer
CORPUS_PATH = Path(r'G:\Projects\Python\OnyxAI\Corpus\TinyStoriesV2-valid.txt')  # Corpus to tokenize
OUTPUT_PATH = Path(r'G:\Projects\Python\OnyxAI\Corpus\tokenized_valid.npy')  # Output token file
VOCAB_SIZE = 30000  # Must match tokenizer vocab size
CHUNK_SIZE = 10_000_000  # 10 MB chunks


if __name__ == "__main__":
    print("=" * 70)
    print("Corpus Tokenization Script (Chunked Loading)")
    print("=" * 70)
    
    # Verify files exist
    if not TOKENIZER_PATH.exists():
        print(f"\n✗ Error: Tokenizer not found at {TOKENIZER_PATH.absolute()}")
        print("  Train the tokenizer first using train_tokenizer_chunked.py")
        exit(1)
    
    if not CORPUS_PATH.exists():
        print(f"\n✗ Error: Corpus file not found at {CORPUS_PATH.absolute()}")
        exit(1)
    
    # Get file info
    corpus_size = CORPUS_PATH.stat().st_size
    corpus_size_mb = corpus_size / 1024 / 1024
    
    print(f"\n[1] Loading Tokenizer...")
    print(f"    From: {TOKENIZER_PATH.absolute()}")
    
    encoder = BPE(vocab_size=VOCAB_SIZE)
    encoder.load(TOKENIZER_PATH)
    
    print(f"\n[2] Corpus Information:")
    print(f"    File: {CORPUS_PATH.absolute()}")
    print(f"    Size: {corpus_size_mb:.2f} MB ({corpus_size:,} bytes)")
    print(f"    Chunk size: {CHUNK_SIZE / 1024 / 1024:.1f} MB")
    
    print(f"\n[3] Output Configuration:")
    print(f"    Will save to: {OUTPUT_PATH.absolute()}")
    print(f"    Format: numpy memory-mapped uint32 array")
    
    # Tokenize corpus
    print(f"\n[4] Tokenizing corpus...")
    
    start_time = time.time()
    total_tokens = encoder.tokenize_corpus_chunked(
        file_path=CORPUS_PATH,
        output_path=OUTPUT_PATH,
        chunk_size=CHUNK_SIZE,
        show_progress=True
    )
    tokenization_time = time.time() - start_time
    
    # Final statistics
    output_size = OUTPUT_PATH.stat().st_size
    output_size_mb = output_size / 1024 / 1024
    compression_ratio = corpus_size / total_tokens
    
    print(f"\n[5] Tokenization Complete!")
    print(f"    {'─' * 60}")
    print(f"    Time elapsed         : {tokenization_time:.2f} seconds")
    print(f"    Original size        : {corpus_size_mb:.2f} MB")
    print(f"    Token file size      : {output_size_mb:.2f} MB")
    print(f"    Total tokens         : {total_tokens:,}")
    print(f"    Compression ratio    : {compression_ratio:.2f}x")
    print(f"    Tokens per second    : {total_tokens / tokenization_time:,.0f}")
    print(f"    {'─' * 60}")
    
    # Space comparison
    space_saved_mb = corpus_size_mb - output_size_mb
    space_saved_percent = (space_saved_mb / corpus_size_mb) * 100
    
    print(f"\n[6] Space Comparison:")
    print(f"    Original text        : {corpus_size_mb:>12.2f} MB")
    print(f"    Tokenized (uint32)   : {output_size_mb:>12.2f} MB")
    print(f"    Space saved          : {space_saved_mb:>12.2f} MB ({space_saved_percent:.1f}%)")
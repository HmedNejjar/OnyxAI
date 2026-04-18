from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE as HFBpe
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import urllib.request


class BPE:
    """
    Byte-pair encoding (BPE) tokenizer implementation using Hugging Face Tokenizers library.
        
    Features:
        - Chunked corpus loading for memory efficiency
        - Streaming tokenization of large files
        - Optional saving of tokenizer to disk
    
    Attributes:
        vocab_size (int): Total tokens in the final vocabulary
        tokenizer (Tokenizer): Hugging Face Tokenizer instance
    """
    
    def __init__(self, vocab_size: int) -> None:
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size (int): Total tokens in the final vocabulary (includes base 256 bytes + learned merges)
        """
        self.vocab_size = vocab_size
        
        # Initialize Hugging Face BPE tokenizer with byte-level preprocessing
        self.tokenizer = Tokenizer(HFBpe())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of token IDs.
        
        Args:
            text (str): Input string to encode
            
        Returns:
            list[int]: List of token IDs representing the encoded text
        """
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: list) -> str:
        """
        Convert a list of token IDs back to a string.
        
        Args:
            ids (list): List of token IDs to decode
            
        Returns:
            str: Reconstructed string
        """
        return self.tokenizer.decode(ids)

    def _load_chunks(self, file_path: str | Path, chunk_size: int = 10_000_000):
        """
        Generator that yields text chunks from a file, split by whitespace.
        
        This function reads the file in memory-efficient chunks and yields
        complete words/segments to avoid splitting in the middle of tokens.
        
        Args:
            file_path (str | Path): Path to the text file to load
            chunk_size (int): Approximate number of characters per chunk (default: 10MB)
            
        Yields:
            str: Text chunks split intelligently by whitespace
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = ""
            
            while True:
                # Read chunk from file
                chunk = f.read(chunk_size)
                
                if not chunk:
                    # End of file - yield remaining buffer
                    if buffer.strip():
                        yield buffer
                    break
                
                # Add to buffer
                buffer += chunk
                
                # Find last space to avoid splitting words
                last_space_idx = buffer.rfind(' ')
                
                if last_space_idx != -1:
                    # Yield complete text up to last space
                    yield buffer[:last_space_idx]
                    # Keep remainder for next iteration
                    buffer = buffer[last_space_idx + 1:]

    def train_chunked(self, file_path: str | Path, save_path: str | Path | None = None, chunk_size: int = 10000000, show_progress: bool = True) -> int:
        """
        Train tokenizer on a large corpus file by loading it in chunks.
        
        This method is memory-efficient for large corpora by:
        1. Loading the file in chunks separated by whitespace
        2. Training the tokenizer on chunk by chunk
        3. Tracking total tokens compressed from the original text
        
        Args:
            file_path (str | Path): Path to the training corpus file
            save_path (str | Path | None): Optional path to save the trained tokenizer
            chunk_size (int): Approximate chunk size in bytes (default: 10MB)
            show_progress (bool): Whether to print progress information (default: True)
            
        Returns:
            int: Total number of tokens the corpus was compressed into
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {file_path}")
        
        if show_progress:
            print(f"[BPE] Loading corpus from: {file_path}")
            file_size = file_path.stat().st_size
            print(f"[BPE] File size: {file_size / 1024 / 1024:.2f} MB")
            print(f"[BPE] Chunk size: {chunk_size / 1024 / 1024:.2f} MB")
        
        # Configure BPE trainer
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["<unk>", "<pad>"])
        
        # Generator of file chunks
        chunk_generator = self._load_chunks(file_path, chunk_size)
        
        # Train on chunks
        if show_progress:
            print(f"[BPE] Training tokenizer...")
        
        self.tokenizer.train_from_iterator(chunk_generator, trainer=trainer)
        
        # Encode entire corpus and track total tokens
        if show_progress:
            print(f"[BPE] Tokenizing corpus to get compression stats...")
        
        total_tokens = self._count_tokens_chunked(file_path, chunk_size, show_progress)
        
        # Save tokenizer if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save(str(save_path))
            if show_progress:
                print(f"✓ Tokenizer saved to {save_path}")
        
        return total_tokens

    def _count_tokens_chunked(self, file_path: str | Path, chunk_size: int = 10_000_000, 
                              show_progress: bool = True) -> int:
        """
        Count total tokens in a corpus by tokenizing chunks and summing token counts.
        
        Args:
            file_path (str | Path): Path to the corpus file
            chunk_size (int): Chunk size in bytes
            show_progress (bool): Whether to print progress
            
        Returns:
            int: Total token count for the entire corpus
        """
        total_tokens = 0
        chunk_num = 0
        
        for chunk in self._load_chunks(file_path, chunk_size):
            chunk_tokens = len(self.encode(chunk))
            total_tokens += chunk_tokens
            chunk_num += 1
            
            if show_progress and chunk_num % 5 == 0:
                print(f"  [Progress] Chunk {chunk_num}: {chunk_tokens:,} tokens (total: {total_tokens:,})")
        
        return total_tokens

    def tokenize_corpus_chunked(self, file_path: str | Path, output_path: str | Path,
                                chunk_size: int = 10_000_000, show_progress: bool = True) -> int:
        """
        Tokenize a large corpus file and save token IDs to a numpy file.
        
        This method streams through the corpus file, tokenizes chunks, and writes
        token IDs sequentially to a numpy memory-mapped file for efficient storage.
        
        Args:
            file_path (str | Path): Path to the text corpus to tokenize
            output_path (str | Path): Path to save token IDs as .npy file
            chunk_size (int): Approximate chunk size in bytes (default: 10MB)
            show_progress (bool): Whether to print progress information (default: True)
            
        Returns:
            int: Total number of tokens generated from the corpus
        """
        import numpy as np
        
        file_path = Path(file_path)
        output_path = Path(output_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {file_path}")
        
        if show_progress:
            print(f"[BPE] Tokenizing corpus: {file_path}")
            print(f"[BPE] Output will be saved to: {output_path}")
            file_size = file_path.stat().st_size
            print(f"[BPE] File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # First pass: count total tokens needed
        if show_progress:
            print(f"[BPE] Counting total tokens (first pass)...")
        total_tokens = self._count_tokens_chunked(file_path, chunk_size, show_progress=False)
        
        if show_progress:
            print(f"[BPE] Total tokens to write: {total_tokens:,}")
            print(f"[BPE] Tokenizing and writing to disk (second pass)...")
        
        # Second pass: write tokens to file
        # Create memory-mapped numpy array
        token_array = np.memmap(output_path, dtype=np.uint32, mode='w+', shape=(total_tokens,))
        
        current_idx = 0
        chunk_num = 0
        
        for chunk in self._load_chunks(file_path, chunk_size):
            tokens = self.encode(chunk)
            chunk_token_count = len(tokens)
            
            # Write tokens to array
            token_array[current_idx:current_idx + chunk_token_count] = tokens
            current_idx += chunk_token_count
            chunk_num += 1
            
            if show_progress and chunk_num % 5 == 0:
                percent = (current_idx / total_tokens) * 100
                print(f"  [Progress] Chunk {chunk_num}: {chunk_token_count:,} tokens ({percent:.1f}% complete)")
        
        # Flush to disk
        token_array.flush()
        
        if show_progress:
            output_size_mb = output_path.stat().st_size / 1024 / 1024
            compression_ratio = file_path.stat().st_size / total_tokens
            print(f"✓ Tokenization complete!")
            print(f"  Output file size: {output_size_mb:.2f} MB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Total tokens: {total_tokens:,}")
        
        return total_tokens

    def train(self, text: str, save_path: str | Path | None = None) -> int:
        """
        Train tokenizer on text string (for small corpora or testing).
        
        For large corpora, use train_chunked() instead.
        
        Args:
            text (str): Training corpus text
            save_path (str | Path | None): Optional path to save the trained tokenizer
            
        Returns:
            int: Number of tokens the text was compressed into
        """
        # Configure BPE trainer with specified vocabulary size and special tokens
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["<unk>", "<pad>"])
        
        # Train tokenizer on the provided text
        self.tokenizer.train_from_iterator([text], trainer=trainer, length=len(text))
        
        # Save tokenizer if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save(str(save_path))
            print(f"✓ Tokenizer saved to {save_path}")
        
        # Return token count
        return len(self.encode(text))

    def load(self, path: str | Path) -> None:
        """
        Load a pre-trained tokenizer from disk.
        
        Args:
            path (str | Path): Path to the saved tokenizer file
        """
        self.tokenizer = Tokenizer.from_file(str(path))
        print(f"✓ Tokenizer loaded from {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("BPE Tokenizer - Chunked Loading Demo")
    print("=" * 70)
    
    # Initialize tokenizer
    vocab_size = 5000
    print(f"\n[1] Initializing BPE tokenizer (vocab_size={vocab_size})...")
    encoder = BPE(vocab_size)
    
    # Download sample text
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    print(f"\n[2] Downloading sample text from Project Gutenberg...")
    print(f"    URL: {url}")
    
    try:
        response = urllib.request.urlopen(url)
        text = response.read().decode('utf-8')[:100000]  # 100KB sample
        print(f"    Downloaded {len(text):,} characters")
        
        # Train and encode
        print(f"\n[3] Training tokenizer on sample text...")
        token_count = encoder.train(text, None)
        
        # Display statistics
        print(f"\n[4] Tokenization Results:")
        print(f"    {'─' * 50}")
        print(f"    Original characters : {len(text):>12,}")
        print(f"    Original bytes      : {len(list(text.encode())):>12,}")
        print(f"    Compressed tokens   : {token_count:>12,}")
        ratio = len(text.encode()) / token_count
        print(f"    Compression ratio   : {ratio:>12.2f}x")
        print(f"    {'─' * 50}")
                
    except Exception as e:
        print(f"    ✗ Error: {e}")
        print("    (Demo skipped - check your internet connection)")
    
    print("\n" + "=" * 70)
    print("For large corpora, use: encoder.train_chunked(file_path)")
    print("To tokenize corpus to disk: encoder.tokenize_corpus_chunked(input, output)")
    print("=" * 70)
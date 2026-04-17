import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE as HFBpe
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import urllib.request


class BPE:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(HFBpe())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids)

    def train(self, text: str, save_path: str | Path | None = None) -> list[int]:
        """Train on text, optionally save to disk. Returns tokenized ids."""
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["<unk>", "<pad>"])
        self.tokenizer.train_from_iterator([text], trainer=trainer, length=len(text))
        
        if save_path:
            self.tokenizer.save(str(save_path))
            print(f"✓ Tokenizer saved to {save_path}")
        
        return self.encode(text)

    def load(self, path: str | Path) -> None:
        """Load tokenizer from disk"""
        self.tokenizer = Tokenizer.from_file(str(path))
        print(f"✓ Tokenizer loaded from {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("BPE Tokenizer Demo")
    print("=" * 60)
    
    # Initialize tokenizer
    vocab_size = 2000
    print(f"\n[1] Initializing BPE tokenizer (vocab_size={vocab_size})...")
    encoder = BPE(vocab_size)
    
    # Download sample text
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    print(f"\n[2] Downloading sample text from Project Gutenberg...")
    print(f"    URL: {url}")
    
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')[:50000]
    print(f"    Downloaded {len(text):,} characters")
    
    # Train and encode
    print(f"\n[3] Training tokenizer on sample text...")
    ids = encoder.train(text, None)
    
    # Display statistics
    print(f"\n[4] Tokenization Results:")
    print(f"    {'─' * 40}")
    print(f"    Original characters : {len(text):>10,}")
    print(f"    Original bytes      : {len(list(text.encode())):>10,}")
    print(f"    Merged tokens       : {len(ids):>10,}")
    print(f"    Compression ratio   : {len(ids) / len(list(text.encode())):>10.2%}")
    print(f"    {'─' * 40}")
    
    # Demo encode/decode
    sample_text = "Hello, world! This is a test."
    print(f"\n[5] Encode/Decode Demo:")
    print(f"    Original: \"{sample_text}\"")
    
    encoded = encoder.encode(sample_text)
    print(f"    Encoded : {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
    
    decoded = encoder.decode(encoded)
    print(f"    Decoded : \"{decoded}\"")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
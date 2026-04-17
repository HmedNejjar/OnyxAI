import numpy as np
from pathlib import Path
from BPE import BPE

VOCAB_SIZE = 22000
PARENT_FOLDER = Path(r'G:\Projects\Python\OnyxAI')
TOKENIZER_PATH = PARENT_FOLDER / Path(r'Tokenizer/merges.json')
TRAIN_TEXT_PATH = PARENT_FOLDER / Path(r'Corpus/TinyStoriesV2-train.txt')
VAL_TEXT_PATH = PARENT_FOLDER / Path(r'Corpus/TinyStoriesV2-valid.txt')
TRAIN_TOKENS_PATH = PARENT_FOLDER / Path(r'Corpus/train_tokens.npy')
VAL_TOKENS_PATH = PARENT_FOLDER / Path(r'Corpus/val_tokens.npy')

def tokenize_and_save(text_path: Path, tokens_path: Path, tokenizer: BPE) -> None:
    """Tokenize text and save as binary numpy file"""
    print(f"Loading text from {text_path}...")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Tokenizing {len(text):,} characters...")
    tokens = tokenizer.encode(text)
    
    print(f"Saving {len(tokens):,} tokens to {tokens_path}...")
    np.save(tokens_path, np.array(tokens, dtype=np.uint32))
    print(f"Done! Compression: {len(text)} chars → {len(tokens)} tokens ({len(text)/len(tokens):.2f}x)\n")

def load_tokenizer(path: Path | str) -> BPE:
    tokenizer = BPE(VOCAB_SIZE)
    print(f"Loading tokenizer from {path}\n")
    tokenizer.merges = tokenizer.load(path)
    return tokenizer

if __name__ == "__main__":
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    # Tokenize training and validation sets
    tokenize_and_save(TRAIN_TEXT_PATH, TRAIN_TOKENS_PATH, tokenizer)
    tokenize_and_save(VAL_TEXT_PATH, VAL_TOKENS_PATH, tokenizer)
    
    print("Tokenization complete!")
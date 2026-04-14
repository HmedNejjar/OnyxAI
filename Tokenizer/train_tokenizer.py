from BPE import BPE
from pathlib import Path

FILE_PATH = Path(r'corpus.txt')
VOCAB_SIZE = 22000

with open(FILE_PATH, 'r') as f:
    text = f.read()
    
print(f"Training on {len(text):,} characters")

# Train
encoder = BPE(vocab_size=VOCAB_SIZE)
merges, ids = encoder.train(text)

# Save
encoder.save(merges, 'merges.json')

print(f"Vocabulary size: {len(merges) + 256}")
print(f"Compression: {len(text.encode())} → {len(ids)} tokens ({len(text.encode())/len(ids):.2f}x)")
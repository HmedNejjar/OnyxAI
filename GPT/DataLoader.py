import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(1, "G:\\Projects\\Python\\OnyxAI")
from Tokenizer.BPE import BPE


class GPTDataset(Dataset):
    """
    Dataset for preparing autoregressive language-model training samples.

    The dataset loads pre-tokenized data from a numpy memory-mapped file
    and exposes sliding context/target pairs suitable for next-token
    prediction training.

    Args:
        text (str): Raw training text (kept for API compatibility).
        tokens_path (str | Path): Path to numpy file containing pre-tokenized
            token ids saved in .npy format.
        context_size (int): Number of tokens to use as the input context
            for each training example.

    Returns:
        GPTDataset: A dataset instance that yields `(context, target)` pairs
        from the tokenized data.
    """
    def __init__(self, text: str, tokens_path: str | Path, context_size: int) -> None:
        """
        Build a dataset from pre-tokenized numpy data.

        Args:
            text (str): Raw text (unused, kept for API compatibility).
            tokens_path (str | Path): Path to the .npy file containing
                pre-computed token ids.
            context_size (int): Number of tokens in each input context window.

        Returns:
            None: This method initializes the dataset in place.
        """
        super().__init__()
        self.tokens = np.load(tokens_path, mmap_mode='r')
        self.context_size = context_size
        
    def __len__(self) -> int:
        """
        Return the number of available training samples in the dataset.

        The number of samples is the total token count minus the context
        window size and one additional token for the target.

        Args:
            None

        Returns:
            int: Total number of training samples that can be generated.
        """
        return len(self.tokens) - self.context_size - 1
    
    def __getitem__(self, idx) -> tuple:
        """
        Return one training sample at the requested index.

        The input context is a window of tokens starting at ``idx``.
        The target is the same window shifted by one position so the model
        learns to predict the next token at every step.

        Args:
            idx (int): Starting token index for the training sample.

        Returns:
            tuple: A pair ``(context, target)`` where both elements are
            ``torch.long`` tensors.
        """
        # Clamp index to valid range to prevent out-of-bounds access
        max_start_idx = len(self.tokens) - self.context_size - 1
        idx = min(idx, max_start_idx)

        # Context tokens are the current sliding window
        context = torch.tensor(self.tokens[idx:idx + self.context_size], dtype=torch.long)
        # Target tokens are shifted by one position (next-token prediction)
        target = torch.tensor(self.tokens[idx + 1:idx + self.context_size + 1], dtype=torch.long)
        
        return (context, target)

    def get_dataloader(self, text: str, tokens_path: str | Path, context_size: int, batch_size: int) -> DataLoader:
        """
        Create a shuffled dataloader for GPT training.

        Args:
            text (str): Raw text (unused, kept for API compatibility).
            tokens_path (str | Path): Path to pre-tokenized numpy data file.
            context_size (int): Number of tokens in each input context window.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: A PyTorch dataloader that yields batched context and
            target tensors with shuffling enabled.
        """
        dataset = GPTDataset(text, tokens_path, context_size)
        # Shuffle samples so the model sees varied training batches each epoch
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


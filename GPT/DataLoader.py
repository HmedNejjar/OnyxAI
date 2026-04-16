import sys
import torch
from torch.utils.data import Dataset,  DataLoader
from Tokenizer.BPE import BPE

sys.path.insert(1, "G:\\Projects\\Python\\OnyxAI\\Tokenizer\\BPE.py")

class GPTDataset(Dataset):
    """
    Dataset for preparing autoregressive language-model training samples.

    The dataset tokenizes the full input text once, then exposes sliding
    context/target pairs suitable for next-token prediction training.

    Args:
        text (str): Raw training text to tokenize and split into samples.
        tokenizer (BPE): Tokenizer used to convert text into token ids.
        context_size (int): Number of tokens to use as the input context
            for each training example.

    Returns:
        GPTDataset: A dataset instance that yields `(context, target)` pairs
        from the tokenized text.
    """
    def __init__(self, text:str, tokenizer:BPE, context_size: int) -> None:
        """
        Build a dataset from raw text and a tokenizer.

        Args:
            text (str): Raw text that will be tokenized into training data.
            tokenizer (BPE): Tokenizer instance used to encode the text.
            context_size (int): Number of tokens in each input context window.

        Returns:
            None: This method initializes the dataset in place.
        """
        super().__init__()
        self.tokenzier = tokenizer
        self.context_size = context_size
        
        # Tokenize the full corpus once so indexing stays fast during training.
        self.tokens = tokenizer.encode(text)
        
    def __len__(self) -> int:
        """
        Return the number of available tokens in the dataset.

        Args:
            None

        Returns:
            int: Total number of token ids stored in the dataset.
        """
        return len(self.tokens)
    
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
         # Ensure we don't go out of bounds
        max_start_idx = len(self.tokens) - self.context_size - 1
        idx = min(idx, max_start_idx)

        # Context tokens are the current sliding window
        context = torch.tensor(self.tokens[idx:idx + self.context_size], dtype=torch.long)
        # Target tokens are shifted by one position
        target = torch.tensor(self.tokens[idx + 1:idx + self.context_size + 1], dtype=torch.long)
        
        return (context, target)

    def get_dataloader(self, text:str, tokenizer:BPE, context_size:int, batch_size: int) -> DataLoader:
        """
        Create a shuffled dataloader for GPT training.

        Args:
            text (str): Raw text used to construct the dataset.
            tokenizer (BPE): Tokenizer used to encode the text.
            context_size (int): Number of tokens in each input context window.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: A PyTorch dataloader that yields batched context and
            target tensors.
        """
        dataset = GPTDataset(text, tokenizer, context_size)
        # Shuffle samples so the model sees varied training batches each epoch.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader

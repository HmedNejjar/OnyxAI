import json
from pathlib import Path
import urllib.request


class BPE:
    """Byte-pair encoding tokenizer — learns to compress text into the fewest tokens possible."""

    def __init__(self, vocab_size: int) -> None:
        """
        Args:
            vocab_size: Total tokens in the final vocabulary (base 256 bytes + learned merges).
        """
        self.vocab_size = vocab_size
        self.merges = {}

    def encode(self, text: str) -> list[int]:
        """
        Turn text into a list of token ids.
        Starts from raw bytes, then applies each learned merge in the order it was learned.

        Args:
            text: Input string to encode.
            merges: Merge rules learned during training — maps byte pairs to their new token id.

        Returns:
            List of token ids representing the encoded text.
        """
        ids = list(text.encode())
        for pair, merged_id in sorted(self.merges.items(), key=lambda x: x[1]):
            ids = self.merge(ids, pair, merged_id)
        return ids

    def merge(self, ids: list, pair: tuple, merge_id: int) -> list:
        """
        Scan through ids and replace every adjacent occurrence of pair with merge_id.
        When a match is found, consume both elements and emit one — so i jumps by 2.

        Args:
            ids: Current list of token ids.
            pair: The two-token sequence to replace.
            merge_id: The new token id that replaces the pair.

        Returns:
            New list with all occurrences of pair collapsed into merge_id.
        """
        result, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(merge_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def adjacent_pairs(self, ids: list) -> dict[tuple, int]:
        """
        Count how often each adjacent pair of token ids appears.
        The most frequent pair is the next candidate for merging.

        Args:
            ids: Current list of token ids.

        Returns:
            Dict mapping each (left, right) pair to how many times it appears.
        """
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def reverse_merge(self) -> dict[int, tuple]:
        """
        Flip the merge table so decoding can expand a merged id back into its two parts.

        Args:
            merges: Merge rules from training — maps byte pairs to merged token ids.

        Returns:
            Reversed table — maps each merged token id back to its (left, right) pair.
        """
        return {new_id: pair for pair, new_id in self.merges.items()}

    def _decode_token(self, id: int, reversed_merges: dict[int, tuple]) -> bytes:
        """
        Recursively expand a single token id down to its raw bytes.
        Base case: ids 0-255 map directly to characters via chr().
        Recursive case: ids 256+ are merged tokens — unpack the pair and decode both halves.

        Args:
            id: Token id to expand.
            reversed_merges: Flipped merge table from reverse_merge().

        Returns:
            The string that this token represents.
        """
        if id < 256:
            return bytes([id])
        pair = reversed_merges[id]
        return self._decode_token(pair[0], reversed_merges) + self._decode_token(pair[1], reversed_merges)

    def train(self, text: str) -> tuple[dict[tuple, int], list[int]]:
        """
        Learn merge rules from text until the vocabulary reaches vocab_size.
        Each iteration: find the most frequent pair, merge it everywhere, record the rule.

        Args:
            text: Training corpus to learn from.

        Returns:
            merges: All learned merge rules in the order they were learned.
            ids: The fully merged token sequence of the training text.
        """
        ids = self.encode(text)
        for i in range(self.vocab_size - 256):
            new_id = 256 + i
            counts = self.adjacent_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=lambda pair: counts[pair])
            ids = self.merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
        return self.merges, ids

    def decode(self, ids: list) -> str:
        """
        Convert a list of token ids back into a string.

        Args:
            ids: Token ids to decode.
            reversed_merges: Flipped merge table from reverse_merge().

        Returns:
            Reconstructed string.
        """
        reversed_merges = self.reverse_merge()
        byte_seq = b"".join(self._decode_token(token_id, reversed_merges) for token_id in ids)
        return byte_seq.decode("utf-8")

    def save(self, merges: dict[tuple, int], path: str | Path) -> None:
        """
        Save merge rules to disk as JSON.
        Pairs are saved sorted by their merge id (i.e. in the order they were learned),
        so load() can reconstruct the ids correctly by position.

        Args:
            merges: Merge rules from training.
            path: Where to save the JSON file.
        """
        path = Path(path)
        merges_list = [[pair[0], pair[1]] for pair, _ in sorted(merges.items(), key=lambda x: x[1])]
        with open(path, 'w') as f:
            json.dump(merges_list, f)

    def load(self, path: str | Path) -> dict[tuple, int]:
        """
        Load merge rules from disk and rebuild their token ids by position.
        Position 0 → id 256, position 1 → id 257, and so on.

        Args:
            path: JSON file saved by save().

        Returns:
            Merge rules ready to pass into encode() or reverse_merge().
        """
        path = Path(path)
        with open(path, 'r') as f:
            merges_list = json.load(f)
        return {(pair[0], pair[1]): 256 + i for i, pair in enumerate(merges_list)}


if __name__ == "__main__":

    encoder = BPE(2000)

    url = "https://www.gutenberg.org/files/100/100-0.txt"
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')[:50000]

    merges, ids = encoder.train(text)
    print(f"Original bytes: {len(list(text.encode()))}")
    print(f"Merged tokens:  {len(ids)}")
    print(f"Compression:    {len(ids) / len(list(text.encode())):.2%}")
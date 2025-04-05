import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import csv
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, seq_length: int):
        self.sequences = sequences
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.sequences) - self.seq_length
        
    def __getitem__(self, idx):
        return (self.sequences[idx:idx + self.seq_length],
                self.sequences[idx + 1:idx + self.seq_length + 1])

class DataPreparator:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = set()
        
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from list of texts."""
        # Count word frequencies
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter words by minimum frequency and sort by frequency
        filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= min_freq]
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        
        # Take top vocab_size-2 words (leaving room for special tokens)
        top_words = [word for word, _ in sorted_words[:self.vocab_size-2]]
        
        # Create vocabulary set with special tokens
        self.vocab = set(top_words)
        self.vocab.add('<PAD>')
        self.vocab.add('<UNK>')
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(self.vocab))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices."""
        words = text.split()
        return [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text."""
        return ' '.join(self.idx_to_word[idx] for idx in sequence)

def load_text_data(
    file_path: str,
    file_type: str = 'txt',
    max_samples: Optional[int] = None
) -> List[str]:
    """Load text data from various file formats."""
    texts = []
    
    if file_type == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif file_type == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assuming JSON is a list of objects with a 'text' field
            texts = [item['text'] for item in data]
    elif file_type == 'csv':
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Assuming CSV has a 'text' column
            texts = [row['text'] for row in reader]
    
    if max_samples:
        texts = texts[:max_samples]
    
    return texts

def prepare_data(
    file_path: Optional[str] = None,
    texts: Optional[List[str]] = None,
    file_type: str = 'txt',
    vocab_size: int = 50257,
    seq_length: int = 50,
    max_samples: Optional[int] = None,
    min_freq: int = 2
) -> Tuple[TextDataset, DataPreparator]:
    """Prepare data for training.
    
    Args:
        file_path: Path to the data file (optional if texts is provided)
        texts: List of texts to process (optional if file_path is provided)
        file_type: Type of input file ('txt', 'json', 'csv')
        vocab_size: Size of the vocabulary
        seq_length: Length of sequences for training
        max_samples: Maximum number of samples to use
        min_freq: Minimum word frequency for vocabulary
    """
    # Get texts either from file or direct input
    if texts is None and file_path is not None:
        texts = load_text_data(file_path, file_type, max_samples)
    elif texts is not None:
        if max_samples:
            texts = texts[:max_samples]
    else:
        raise ValueError("Either file_path or texts must be provided")
    
    # Initialize data preparator and build vocabulary
    preparator = DataPreparator(vocab_size)
    preparator.build_vocabulary(texts, min_freq)
    
    # Convert all texts to sequences
    sequences = []
    for text in tqdm(texts, desc="Converting texts to sequences"):
        sequences.extend(preparator.text_to_sequence(text))
    
    # Convert to tensor
    sequences = torch.tensor(sequences)
    
    # Create dataset
    dataset = TextDataset(sequences, seq_length)
    
    return dataset, preparator

if __name__ == "__main__":
    # Example usage
    file_path = "path/to/your/data.txt"
    dataset, preparator = prepare_data(
        file_path=file_path,
        vocab_size=50257,
        seq_length=50,
        max_samples=10000
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(preparator.vocab)}")
    
    # Example of converting between text and sequence
    sample_text = "This is a sample text"
    sequence = preparator.text_to_sequence(sample_text)
    reconstructed_text = preparator.sequence_to_text(sequence)
    
    print(f"Original text: {sample_text}")
    print(f"Reconstructed text: {reconstructed_text}")
    print(f"Texts match: {sample_text == reconstructed_text}") 
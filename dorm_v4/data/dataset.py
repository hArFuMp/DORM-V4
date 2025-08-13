import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizerFast
import numpy as np
import os

class PretokenizedDataset(Dataset):
    """
    Efficient dataset class for reading pre-tokenized binary files.
    """
    def __init__(self, file_path, block_size):
        """
        Args:
            file_path (str): Path to the .bin file.
            block_size (int): Max sequence length for model input.
        """
        self.block_size = block_size
        # Load data as a uint16 memory-map
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')

    def __len__(self):
        # Dataset length is the number of blocks
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        """
        Returns an item (x, y) from the dataset at a specific index.
        """
        # Extract chunk for the given index
        start_index = idx * self.block_size
        end_index = start_index + self.block_size
        chunk = torch.from_numpy(self.data[start_index:end_index].astype(np.int64))
        
        # Create input (x) and target (y) (y is x shifted by one)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_dataloader(data_config, use_pretokenized=True):
    """Creates DataLoader for training."""
    if use_pretokenized:
        data_dir = os.path.dirname(data_config.train_file) # Directory containing .parquet files
        train_bin_path = os.path.join(data_dir, 'train.bin')
        eval_bin_path = os.path.join(data_dir, 'val.bin')

        if not os.path.exists(train_bin_path) or not os.path.exists(eval_bin_path):
            raise FileNotFoundError(
                f"'{train_bin_path}' or '{eval_bin_path}' not found. "
                f"Please run `preprocess.py` first to generate .bin files."
            )

        train_dataset = PretokenizedDataset(train_bin_path, data_config.block_size)
        eval_dataset = PretokenizedDataset(eval_bin_path, data_config.block_size)
    else:
        # ParquetDataset logic (not fully implemented in this refactored version)
        raise NotImplementedError("ParquetDataset is not fully supported in this refactored version. Please run preprocess.py first.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, eval_dataloader

if __name__ == '__main__':
    # Example usage
    # Run preprocess.py first to generate data/train.bin and data/val.bin
    
    @dataclass
    class DummyDataConfig:
        train_file: str = '../data/train-00000-of-00004.parquet' # Path for reference
        eval_file: str = '../data/test-00000-of-00001.parquet'
        block_size: int = 128
        batch_size: int = 32
        num_workers: int = 0

    config = DummyDataConfig()
    
    # Assuming preprocess.py has been run
    print("Creating Dataloader... (preprocess.py must be run first)")
    try:
        train_loader, val_loader = get_dataloader(config, use_pretokenized=True)
        
        print("\nTrain Dataloader check:")
        for x, y in train_loader:
            print(f"x shape: {x.shape}") # Expected: (batch_size, block_size - 1)
            print(f"y shape: {y.shape}") # Expected: (batch_size, block_size - 1)
            break

        print("\nValidation Dataloader check:")
        for x, y in val_loader:
            print(f"x shape: {x.shape}")
            print(f"y shape: {y.shape}")
            break

    except FileNotFoundError as e:
        print(f"\nError: {e}")

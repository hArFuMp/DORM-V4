import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizerFast
import numpy as np
import os

class PretokenizedDataset(Dataset):
    """
    미리 토큰화된 바이너리 파일을 읽어오는 효율적인 데이터셋 클래스.
    데이터를 메모리 매핑하여 사용하므로, 대용량 데이터도 빠르게 처리할 수 있습니다.
    """
    def __init__(self, file_path, block_size):
        """
        Args:
            file_path (str): .bin 파일 경로.
            block_size (int): 모델에 입력될 시퀀스의 최대 길이.
        """
        self.block_size = block_size
        # 데이터를 uint16 타입의 memory-map으로 로드
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')

    def __len__(self):
        # 전체 데이터 길이를 블록 사이즈로 나눈 몫만큼이 데이터셋의 길이가 됨
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        """
        데이터셋에서 특정 인덱스의 아이템(x, y)을 반환합니다.
        """
        # 데이터에서 해당 인덱스의 블록을 잘라냄
        start_index = idx * self.block_size
        end_index = start_index + self.block_size
        chunk = torch.from_numpy(self.data[start_index:end_index].astype(np.int64))
        
        # 입력(x)과 타겟(y)을 생성 (타겟은 입력을 한 칸씩 민 것)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_dataloader(data_config, use_pretokenized=True):
    """
    학습에 사용할 DataLoader를 생성합니다.

    Args:
        data_config (DataConfig): 데이터 관련 설정을 담은 객체.
        use_pretokenized (bool): 미리 토큰화된 .bin 파일을 사용할지 여부.

    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    if use_pretokenized:
        data_dir = os.path.dirname(data_config.train_file) # .parquet 파일이 있는 디렉토리
        train_bin_path = os.path.join(data_dir, 'train.bin')
        eval_bin_path = os.path.join(data_dir, 'val.bin')

        if not os.path.exists(train_bin_path) or not os.path.exists(eval_bin_path):
            raise FileNotFoundError(
                f"'{train_bin_path}' 또는 '{eval_bin_path}'를 찾을 수 없습니다. "
                f"먼저 `preprocess.py`를 실행하여 .bin 파일을 생성하세요."
            )

        train_dataset = PretokenizedDataset(train_bin_path, data_config.block_size)
        eval_dataset = PretokenizedDataset(eval_bin_path, data_config.block_size)
    else:
        # 기존 ParquetDataset 사용 로직 (현재는 구현되지 않음, 필요시 추가)
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
    # --- 사용 예시 ---
    # 이 파일을 직접 실행하기 전에, preprocess.py를 먼저 실행하여
    # data/train.bin과 data/val.bin 파일을 생성해야 합니다.
    
    @dataclass
    class DummyDataConfig:
        train_file: str = '../data/train-00000-of-00004.parquet' # 경로 참고용
        eval_file: str = '../data/test-00000-of-00001.parquet'
        block_size: int = 128
        batch_size: int = 32
        num_workers: int = 0

    config = DummyDataConfig()
    
    # preprocess.py를 실행했다고 가정
    print("Dataloader를 생성합니다... (preprocess.py가 먼저 실행되어야 합니다)")
    try:
        train_loader, val_loader = get_dataloader(config, use_pretokenized=True)
        
        print("\nTrain Dataloader 작동 확인:")
        for x, y in train_loader:
            print(f"x shape: {x.shape}") # 예상: (batch_size, block_size - 1)
            print(f"y shape: {y.shape}") # 예상: (batch_size, block_size - 1)
            break

        print("\nValidation Dataloader 작동 확인:")
        for x, y in val_loader:
            print(f"x shape: {x.shape}")
            print(f"y shape: {y.shape}")
            break

    except FileNotFoundError as e:
        print(f"\n오류: {e}")
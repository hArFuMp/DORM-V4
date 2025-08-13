import os
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# 데이터셋이 있는 폴더 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Parquet 파일 목록
TRAIN_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if 'train' in f and f.endswith('.parquet')]
VALID_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if 'test' in f and f.endswith('.parquet')]

# 토크나이저 파일 경로 (main.py와 동일한 위치를 바라보도록 상대 경로 설정)
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'tokenizer.json')

def process_and_save(files, output_filename):
    """
    Parquet 파일들을 읽어 토큰화하고, 결과를 .bin 파일로 저장합니다.
    """
    print(f"{output_filename} 생성을 시작합니다...")
    all_tokens = []
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

    for file_path in tqdm(files, desc=f"Processing {output_filename}"):
        df = pd.read_parquet(file_path)
        # 'text' 컬럼이 있다고 가정합니다. 실제 데이터에 맞게 수정이 필요할 수 있습니다.
        if 'text' in df.columns:
            text_data = df['text'].tolist()
            # 모든 텍스트를 하나의 큰 문자열로 결합
            full_text = '\n'.join(text_data)
            # 토큰화
            tokens = tokenizer.encode(full_text)
            all_tokens.extend(tokens)

    # NumPy 배열로 변환
    arr = np.array(all_tokens, dtype=np.uint16) # GPT-2 토크나이저는 65535 vocab size 미만이므로 uint16으로 충분

    # .bin 파일로 저장
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, 'wb') as f:
        f.write(arr.tobytes())
    
    print(f"{output_filename} 생성이 완료되었습니다. 총 {len(arr)}개의 토큰이 저장되었습니다.")

if __name__ == '__main__':
    if not os.path.exists(TOKENIZER_PATH):
        print(f"오류: 토크나이저 파일({TOKENIZER_PATH})을 찾을 수 없습니다.")
        print("프로젝트 루트에 tokenizer.json 파일이 있는지 확인하세요.")
    else:
        process_and_save(TRAIN_FILES, 'train.bin')
        process_and_save(VALID_FILES, 'val.bin')

from dataclasses import dataclass, field
from transformers import PreTrainedTokenizerFast

@dataclass
class ModelConfig:
    """ Slot-based Transformer 모델 관련 설정 (경량화 버전) """
    vocab_size: int = 51200  # 어휘 사전 크기 (예: KoGPT2)
    n_positions: int = 256   # 최대 시퀀스 길이 (512→256으로 절반 감소)
    n_embd: int = 384        # 임베딩 차원 (768→384로 절반 감소)
    n_layer: int = 6         # 전체 레이어(Slot) 수 (12→6으로 절반 감소)
    n_head: int = 6          # 어텐션 헤드 수 (12→6으로 절반 감소)
    n_inner: int = 1536      # Feed-forward 네트워크 내부 차원 (3072→1536로 절반 감소)
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-6  # LayerNorm ε을 1e-6으로 조정 (안정성 향상)
    initializer_range: float = 0.02

@dataclass
class DataConfig:
    """ 데이터셋 및 DataLoader 관련 설정 """
    train_file: str = "DORM-V4/dorm_v4/data/train-00000-of-00004.parquet"  # 학습 데이터 파일 경로
    eval_file: str = "DORM-V4/dorm_v4/data/test-00000-of-00001.parquet"    # 평가 데이터 파일 경로
    block_size: int = 256                   # 모델 입력 시퀀스 길이 (512→256으로 절반 감소)
    batch_size: int = 32                   # 배치 사이즈 증가 (16→32, 경량화로 인한 여유)
    num_workers: int = 8                    # 데이터 로딩에 사용할 CPU 코어 수 (4→8로 확대)
    pin_memory: bool = True                 # DataLoader pin_memory 옵션
    cache_dataset: bool = True              # 전체 데이터셋 메모리 캐싱 활성화

@dataclass
class SchedulerConfig:
    """ Q-learning 스케줄러 관련 설정 """
    num_actions: int = 3  # 행동(Action)의 개수 (예: 12개 slot을 4개씩 3그룹으로)
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_decay: float = 0.995

@dataclass
class TrainingConfig:
    """ 학습 파이프라인 관련 설정 """
    output_dir: str = "checkpoints/"      # 모델 체크포인트 저장 경로
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    optimizer: str = "AdamW"              # 사용할 옵티마이저
    weight_decay: float = 0.01
    
    # 동적 정밀도 관련 설정
    use_dynamic_precision: bool = True
    fp16_confidence_threshold: float = 0.95 # 이 confidence 이상이면 FP16 사용
    fp16_fixed_epochs: int = 2              # FP16 고정 에폭 수
    grad_accum_steps: int = 2               # gradient accumulation steps (1→2로 증가)
    grad_clip_norm: float = 1.0             # gradient clipping norm
    gradscaler_init_scale: float = 2**10    # GradScaler 초기값 (2**16→2**10으로 축소)
    gradscaler_growth_factor: float = 2.0
    gradscaler_backoff_factor: float = 0.5
    gradscaler_growth_interval: int = 1000  # growth_interval 단축 (2000→1000)
    
    # 체크포인트/슬롯 주기
    checkpoint_interval: int = 1           # 몇 epoch마다 저장
    slot_opt_interval: int = 5             # 몇 epoch마다 slot fusion/pruning
    
    # 러닝레이트 스케줄러
    lr_scheduler_type: str = "cosine"       # linear → cosine으로 변경
    lr_warmup_steps: int = 1000             # warmup steps 증가 (500→1000)
    lr_decay_steps: int = 10000             # decay steps 추가
    
    # 로깅 관련 설정
    logging_interval: int = 10              # N 스텝마다 로그 기록
    wandb_project: str = "DORM-V4-Training"
    wandb_run_name: str = "pretraining-run-1"

@dataclass
class ProjectConfig:
    """ 프로젝트 전체 설정을 통합 관리 """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# config.py를 직접 실행하여 설정 값들을 확인할 수 있습니다.
if __name__ == '__main__':
    config = ProjectConfig()
    
    print("--- Model Configuration ---")
    print(config.model)
    print("\n--- Data Configuration ---")
    print(config.data)
    print("\n--- Scheduler Configuration ---")
    print(config.scheduler)
    print("\n--- Training Configuration ---")
    print(config.training)

# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="D:/민이/spm_en.model",  # 또는 spm_en.model의 경로
#     vocab_file="D:/민이/spm_en.vocab",    # (옵션) vocab 파일 경로
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     bos_token="[BOS]",
#     eos_token="[EOS]"
# )

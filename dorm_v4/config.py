from dataclasses import dataclass, field
from transformers import PreTrainedTokenizerFast

@dataclass
class ModelConfig:
    """Model configuration for Slot-based Transformer."""
    vocab_size: int = 51200  # Vocabulary size (e.g., KoGPT2)
    n_positions: int = 256   # Max sequence length
    n_embd: int = 384        # Embedding dimension
    n_layer: int = 6         # Total number of layers (slots)
    n_head: int = 6          # Number of attention heads
    n_inner: int = 1536      # Inner dimension of Feed-forward network
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-6  # LayerNorm epsilon
    initializer_range: float = 0.02

@dataclass
class DataConfig:
    """Dataset and DataLoader configuration."""
    train_file: str = "DORM-V4/dorm_v4/data/train-00000-of-00004.parquet"  # Path to training data
    eval_file: str = "DORM-V4/dorm_v4/data/test-00000-of-00001.parquet"    # Path to evaluation data
    block_size: int = 256                   # Model input sequence length
    batch_size: int = 32                   # Batch size
    num_workers: int = 8                    # Number of CPU cores for data loading
    pin_memory: bool = True                 # DataLoader pin_memory option
    cache_dataset: bool = True              # Cache entire dataset in memory

@dataclass
class SchedulerConfig:
    """Q-learning scheduler configuration."""
    num_actions: int = 3  # Number of actions
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_decay: float = 0.995

@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    output_dir: str = "checkpoints/"      # Path to save model checkpoints
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    optimizer: str = "AdamW"              # Optimizer to use
    weight_decay: float = 0.01
    
    # Dynamic precision settings
    use_dynamic_precision: bool = True
    fp16_confidence_threshold: float = 0.95 
    fp16_fixed_epochs: int = 2              
    grad_accum_steps: int = 2               # Gradient accumulation steps
    grad_clip_norm: float = 1.0             # Gradient clipping norm
    gradscaler_init_scale: float = 2**10    # GradScaler initial scale
    gradscaler_growth_factor: float = 2.0
    gradscaler_backoff_factor: float = 0.5
    gradscaler_growth_interval: int = 1000  # GradScaler growth interval
    
    # Checkpoint/slot frequency
    checkpoint_interval: int = 1           # Save checkpoint every N epochs
    slot_opt_interval: int = 5             # Slot fusion/pruning interval (epochs)
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"       
    lr_warmup_steps: int = 1000             # Warmup steps
    lr_decay_steps: int = 10000             # Decay steps
    
    # Logging settings
    logging_interval: int = 10              # Log every N steps
    wandb_project: str = "DORM-V4-Training"
    wandb_run_name: str = "pretraining-run-1"

@dataclass
class ProjectConfig:
    """Integrates all project configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# Example of how to access configurations when running config.py directly
if __name__ == '__main__':
    config = ProjectConfig()
    
    print("---", "Model Configuration", "---")
    print(config.model)
    print("\n---", "Data Configuration", "---")
    print(config.data)
    print("\n---", "Scheduler Configuration", "---")
    print(config.scheduler)
    print("\n---", "Training Configuration", "---")
    print(config.training)
"""
DORM-V4: Efficient Pre-training for Large-scale Transformer Models

A slot-based transformer pretraining optimization framework for low-resource GPU environments.
"""

__version__ = "1.0.0"
__author__ = "DORM-V4 Team"
__email__ = "dorm-v4@example.com"

# Core modules
from .config import ProjectConfig, ModelConfig, DataConfig, TrainingConfig, SchedulerConfig
from .model.slot_transformer import SlotTransformer
from .scheduler.q_learning import QLearningScheduler
from .scheduler.curriculum import choose_slot_by_curriculum

# Optimization library
from .utils.optimizer import (
    FastAttention,
    MemoryOptimizer,
    AdvancedGradScaler,
    PerformanceProfiler,
    DataLoaderOptimizer,
    LearningRateScheduler,
    ModelOptimizer,
    create_optimized_training_loop,
)

# Advanced optimization library (DORM-V4 핵심 혁신 기법) - v2로 변경
from .utils.advanced_optimizer_v2 import (
    DORMV4AdvancedOptimizer,
)

# Data utilities
from .data.dataset import ParquetDataset, get_dataloader

# Utility functions
from .utils.logger import setup_wandb

# Main training function
from .main import main

__all__ = [
    # Core
    "ProjectConfig",
    "ModelConfig", 
    "DataConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "SlotTransformer",
    "QLearningScheduler",
    "choose_slot_by_curriculum",
    
    # Optimization
    "FastAttention",
    "MemoryOptimizer", 
    "AdvancedGradScaler",
    "PerformanceProfiler",
    "DataLoaderOptimizer",
    "LearningRateScheduler",
    "ModelOptimizer",
    "create_optimized_training_loop",
    
    # Advanced Optimization (DORM-V4 핵심 혁신 기법) - v2 사용
    "DORMV4AdvancedOptimizer",
    # "AdaptiveBatchSizing",
    # "SlotFusionScheduler", 
    # "MetaSlotCalibration",
    # "CurriculumBatchedPretraining",
    # "SparsePriorRegularization",
    # "MemoryAwareMixedPrecision",
    # "SlotScheduler",
    # "RoundRobinSlotScheduler",
    # "AdaptiveSlotScheduler",
    # "PrecisionController",
    # "CurriculumSelfTraining",
    # "SlotConfig",
    
    # Data
    "ParquetDataset",
    "get_dataloader",
    
    # Utils
    "setup_wandb",
    "main",
]

# Package metadata
__package_info__ = {
    "name": "dorm-v4",
    "version": __version__,
    "description": "Efficient Pre-training for Large-scale Transformer Models",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/your-username/dorm-v4",
    "license": "MIT",
    "python_requires": ">=3.8",
    "keywords": [
        "transformer",
        "pretraining",
        "slot-based", 
        "optimization",
        "deep-learning",
        "nlp",
        "machine-learning",
        "pytorch",
    ],
}
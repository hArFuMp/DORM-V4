"""
DORM-V4 핵심 혁신 기법 구현 (v4 - PyTorch 네이티브 메모리 모니터링)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =====================================================================================
# 1. GPU 백엔드 추상화 계층
# =====================================================================================

class GpuBackend(ABC):
    @abstractmethod
    def get_memory_usage(self) -> float:
        """현재 GPU 메모리 사용률을 0.0에서 1.0 사이의 값으로 반환합니다."""
        pass

class PyTorchGpuBackend(GpuBackend):
    """PyTorch 내장 함수를 사용하는 GPU 백엔드 구현체 (CUDA, ROCm 호환)"""
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.total_memory = 0
        if self.device.type != 'cpu':
            try:
                self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            except Exception as e:
                warnings.warn(f"GPU 장치 속성을 가져오는 데 실패했습니다: {e}")

    def get_memory_usage(self) -> float:
        """PyTorch 내장 함수를 통해 현재 GPU 메모리 사용률을 가져옵니다."""
        if self.device.type == 'cpu' or self.total_memory == 0:
            return 0.0
        
        try:
            allocated = torch.cuda.memory_allocated(self.device)
            return allocated / self.total_memory
        except Exception as e:
            warnings.warn(f"GPU 메모리 사용량을 가져오는 중 오류 발생: {e}")
            return 0.0

# =====================================================================================
# 2. DORM-V4 핵심 최적화 모듈
# =====================================================================================

class SparsePriorRegularization:
    def __init__(self, num_slots: int, l1_lambda: float = 1e-4, **kwargs):
        self.num_slots = num_slots
        self.l1_lambda = l1_lambda
        self.slot_weights = nn.Parameter(torch.ones(num_slots))

    def compute_sparsity_loss(self) -> torch.Tensor:
        return self.l1_lambda * torch.sum(torch.abs(torch.sigmoid(self.slot_weights)))

    def update_usage_history(self, active_slots: List[int]):
        pass

class MemoryAwareMixedPrecision:
    def __init__(self, backend: GpuBackend, memory_threshold: float = 0.8, **kwargs):
        self.backend = backend
        self.memory_threshold = memory_threshold

    def get_precision_context(self) -> Any:
        if self.backend.get_memory_usage() > self.memory_threshold:
            return autocast()
        return torch.no_grad()

class SlotScheduler(ABC):
    @abstractmethod
    def get_next_slots(self, **kwargs) -> List[int]:
        pass

class RoundRobinSlotScheduler(SlotScheduler):
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
    def get_next_slots(self, epoch: int, step: int, **kwargs) -> List[int]:
        return [(epoch + step) % self.num_slots]

# =====================================================================================
# 3. 통합 최적화 매니저
# =====================================================================================

class DORMV4AdvancedOptimizer:
    def __init__(self, num_slots: int, device: torch.device, config: Dict[str, Any]):
        self.num_slots = num_slots
        self.device = device
        self.config = config
        self.backend = PyTorchGpuBackend(device) # device를 전달

        self.spr = SparsePriorRegularization(num_slots, **config.get('spr', {}))
        self.mamp = MemoryAwareMixedPrecision(self.backend, **config.get('mamp', {}))
        self.slot_scheduler = RoundRobinSlotScheduler(num_slots)

    def get_training_config(self, epoch: int, step: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        active_slots_indices = self.slot_scheduler.get_next_slots(epoch=epoch, step=step, metrics=metrics)
        precision_context = self.mamp.get_precision_context()

        # 활성 슬롯 인덱스 리스트를 boolean 마스크 텐서로 변환
        active_slots_mask = torch.zeros(self.num_slots, dtype=torch.bool, device=self.device)
        if active_slots_indices:
            active_slots_mask[active_slots_indices] = True
        
        return {
            'active_slots': active_slots_mask, # 리스트 대신 마스크를 반환
            'precision_context': precision_context,
        }

    def update_metrics(self, metrics: Dict[str, float]):
        if 'active_slots' in metrics:
            self.spr.update_usage_history(metrics['active_slots'])

    def get_regularization_loss(self) -> torch.Tensor:
        return self.spr.compute_sparsity_loss()
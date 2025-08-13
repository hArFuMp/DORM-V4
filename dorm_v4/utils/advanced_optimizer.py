"""
DORM-V4 핵심 혁신 기법 구현
Adaptive Batch Sizing, Slot-Fusion Scheduler, Meta-Slot Calibration 등
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time
import math
import random
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
import psutil
import GPUtil
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SlotConfig:
    """개별 슬롯의 메타 설정"""
    slot_id: int
    learning_rate: float
    precision: str  # 'fp32', 'fp16', 'int8'
    drop_rate: float
    fusion_group: Optional[int] = None
    is_active: bool = True

class AdaptiveBatchSizing:
    """
    Adaptive Batch Sizing (ABS)
    GPU 여유 메모리를 실시간 측정해서 배치 크기를 동적으로 조정
    """
    def __init__(self, 
                 initial_batch_size: int = 8,
                 min_batch_size: int = 1,
                 max_batch_size: int = 64,
                 memory_threshold: float = 0.8,
                 adjustment_factor: float = 0.1):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adjustment_factor = adjustment_factor
        self.current_batch_size = initial_batch_size
        self.memory_history = []
        
    def get_gpu_memory_usage(self) -> float:
        """현재 GPU 메모리 사용률 반환"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 첫 번째 GPU
                return gpu.memoryUtil
            return 0.0
        except:
            return 0.0
    
    def adjust_batch_size(self, current_loss: float = None) -> int:
        """메모리 사용률에 따라 배치 크기 조정"""
        memory_usage = self.get_gpu_memory_usage()
        self.memory_history.append(memory_usage)
        
        # 메모리 사용률이 높으면 배치 크기 감소
        if memory_usage > self.memory_threshold:
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adjustment_factor))
            )
            print(f"메모리 사용률 {memory_usage:.2%} - 배치 크기 감소: {self.current_batch_size}")
        
        # 메모리 사용률이 낮고 손실이 안정적이면 배치 크기 증가
        elif (memory_usage < self.memory_threshold * 0.7 and 
              len(self.memory_history) > 10 and
              np.std(self.memory_history[-10:]) < 0.05):
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adjustment_factor))
            )
            print(f"메모리 여유 - 배치 크기 증가: {self.current_batch_size}")
        
        return self.current_batch_size

class SlotFusionScheduler:
    """
    Slot-Fusion Scheduler (SFS)
    인접 슬롯을 묶어서(fusion) 함께 활성화함으로써 gradient flow를 보존하면서 메모리 절감을 극대화
    """
    def __init__(self, 
                 num_slots: int,
                 fusion_size: int = 2,
                 fusion_strategy: str = "adjacent",  # "adjacent", "skip", "random"
                 fusion_probability: float = 0.3):
        self.num_slots = num_slots
        self.fusion_size = fusion_size
        self.fusion_strategy = fusion_strategy
        self.fusion_probability = fusion_probability
        self.fusion_groups = self._create_fusion_groups()
        
    def _create_fusion_groups(self) -> List[List[int]]:
        """퓨전 그룹 생성"""
        groups = []
        
        if self.fusion_strategy == "adjacent":
            # 인접 슬롯들을 묶음
            for i in range(0, self.num_slots, self.fusion_size):
                group = list(range(i, min(i + self.fusion_size, self.num_slots)))
                groups.append(group)
                
        elif self.fusion_strategy == "skip":
            # 건너뛰며 묶음 (예: [0,2,4], [1,3,5])
            for offset in range(self.fusion_size):
                group = list(range(offset, self.num_slots, self.fusion_size))
                if group:
                    groups.append(group)
                    
        elif self.fusion_strategy == "random":
            # 랜덤하게 묶음
            slots = list(range(self.num_slots))
            random.shuffle(slots)
            for i in range(0, len(slots), self.fusion_size):
                group = slots[i:i + self.fusion_size]
                groups.append(group)
        
        return groups
    
    def get_fusion_slots(self, epoch: int, step: int) -> List[int]:
        """퓨전 슬롯 선택"""
        if random.random() < self.fusion_probability:
            # 퓨전 그룹 중 하나 선택
            group_idx = (epoch + step) % len(self.fusion_groups)
            return self.fusion_groups[group_idx]
        else:
            # 단일 슬롯 선택
            return [(epoch + step) % self.num_slots]
    
    def update_fusion_groups(self, performance_metrics: Dict[str, float]):
        """성능 메트릭에 따라 퓨전 그룹 동적 업데이트"""
        # 성능이 좋은 퓨전 패턴을 더 자주 사용하도록 조정
        pass

class MetaSlotCalibration:
    """
    Meta-Slot Calibration (MSC)
    각 슬롯별 학습률·precision·drop-rate 등을 메타러닝으로 주기적으로 자동 최적화
    """
    def __init__(self, 
                 num_slots: int,
                 optimization_interval: int = 100,
                 lr_range: Tuple[float, float] = (1e-6, 1e-3),
                 precision_options: List[str] = ['fp32', 'fp16', 'int8'],
                 drop_rate_range: Tuple[float, float] = (0.0, 0.5)):
        self.num_slots = num_slots
        self.optimization_interval = optimization_interval
        self.lr_range = lr_range
        self.precision_options = precision_options
        self.drop_rate_range = drop_rate_range
        
        # 각 슬롯별 설정 초기화
        self.slot_configs = {}
        for i in range(num_slots):
            self.slot_configs[i] = SlotConfig(
                slot_id=i,
                learning_rate=np.random.uniform(*lr_range),
                precision=random.choice(precision_options),
                drop_rate=np.random.uniform(*drop_rate_range)
            )
        
        self.performance_history = {}
        self.optimization_step = 0
        
    def get_slot_config(self, slot_id: int) -> SlotConfig:
        """슬롯 설정 반환"""
        return self.slot_configs.get(slot_id, self.slot_configs[0])
    
    def update_performance(self, slot_id: int, metrics: Dict[str, float]):
        """슬롯별 성능 메트릭 업데이트"""
        if slot_id not in self.performance_history:
            self.performance_history[slot_id] = []
        self.performance_history[slot_id].append(metrics)
        
        self.optimization_step += 1
        
        # 주기적으로 메타 최적화 수행
        if self.optimization_step % self.optimization_interval == 0:
            self._optimize_slot_configs()
    
    def _optimize_slot_configs(self):
        """메타러닝을 통한 슬롯 설정 최적화"""
        for slot_id in range(self.num_slots):
            if slot_id in self.performance_history and len(self.performance_history[slot_id]) > 5:
                # 최근 성능을 기반으로 설정 조정
                recent_performance = self.performance_history[slot_id][-5:]
                avg_loss = np.mean([p.get('loss', 0) for p in recent_performance])
                
                # 손실이 높으면 학습률 증가, 정밀도 향상
                if avg_loss > 2.0:
                    self.slot_configs[slot_id].learning_rate *= 1.1
                    if self.slot_configs[slot_id].precision == 'int8':
                        self.slot_configs[slot_id].precision = 'fp16'
                    elif self.slot_configs[slot_id].precision == 'fp16':
                        self.slot_configs[slot_id].precision = 'fp32'
                
                # 손실이 낮으면 학습률 감소, 정밀도 낮춤
                elif avg_loss < 0.5:
                    self.slot_configs[slot_id].learning_rate *= 0.9
                    if self.slot_configs[slot_id].precision == 'fp32':
                        self.slot_configs[slot_id].precision = 'fp16'
                
                # 범위 제한
                self.slot_configs[slot_id].learning_rate = np.clip(
                    self.slot_configs[slot_id].learning_rate,
                    *self.lr_range
                )
                
                print(f"Slot {slot_id} 최적화: lr={self.slot_configs[slot_id].learning_rate:.2e}, "
                      f"precision={self.slot_configs[slot_id].precision}")

class CurriculumBatchedPretraining:
    """
    Curriculum-Batched Pretraining (CBP)
    시퀀스 길이와 배치 크기를 단계별로(커리큘럼) 점진 확대하는 로직
    """
    def __init__(self,
                 initial_seq_len: int = 64,
                 final_seq_len: int = 1024,
                 initial_batch_size: int = 4,
                 final_batch_size: int = 32,
                 curriculum_stages: int = 5,
                 warmup_epochs: int = 2):
        self.initial_seq_len = initial_seq_len
        self.final_seq_len = final_seq_len
        self.initial_batch_size = initial_batch_size
        self.final_batch_size = final_batch_size
        self.curriculum_stages = curriculum_stages
        self.warmup_epochs = warmup_epochs
        
        # 커리큘럼 스케줄 생성
        self.seq_len_schedule = np.linspace(initial_seq_len, final_seq_len, curriculum_stages)
        self.batch_size_schedule = np.linspace(initial_batch_size, final_batch_size, curriculum_stages)
        
    def get_curriculum_params(self, epoch: int, total_epochs: int) -> Tuple[int, int]:
        """현재 에폭에 맞는 커리큘럼 파라미터 반환"""
        if epoch < self.warmup_epochs:
            # 워밍업 단계
            stage = 0
        else:
            # 커리큘럼 단계 계산
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            stage = min(int(progress * self.curriculum_stages), self.curriculum_stages - 1)
        
        seq_len = int(self.seq_len_schedule[stage])
        batch_size = int(self.batch_size_schedule[stage])
        
        return seq_len, batch_size
    
    def should_advance_stage(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """다음 커리큘럼 단계로 진행할지 결정"""
        # 성능 기준으로 단계 진행 결정
        if 'loss' in metrics and metrics['loss'] < 1.0:
            return True
        return False

class SparsePriorRegularization:
    """
    Sparse-Prior Regularization (SPR)
    슬롯 활성화 비율에 L0/L1 정규화를 적용해 low-impact 슬롯을 자동 pruning
    """
    def __init__(self,
                 num_slots: int,
                 sparsity_target: float = 0.3,
                 l1_lambda: float = 1e-4,
                 l0_lambda: float = 1e-3,
                 pruning_threshold: float = 0.1):
        self.num_slots = num_slots
        self.sparsity_target = sparsity_target
        self.l1_lambda = l1_lambda
        self.l0_lambda = l0_lambda
        self.pruning_threshold = pruning_threshold
        
        # 슬롯별 활성화 가중치 (학습 가능한 파라미터)
        self.slot_weights = nn.Parameter(torch.ones(num_slots))
        self.slot_usage_history = torch.zeros(num_slots)
        
    def get_slot_importance(self) -> torch.Tensor:
        """슬롯별 중요도 점수 반환"""
        return torch.sigmoid(self.slot_weights)
    
    def compute_sparsity_loss(self) -> torch.Tensor:
        """스파시티 정규화 손실 계산"""
        importance = self.get_slot_importance()
        
        # L1 정규화
        l1_loss = self.l1_lambda * torch.sum(torch.abs(importance))
        
        # L0 정규화 (근사)
        l0_loss = self.l0_lambda * torch.sum(torch.sigmoid(importance * 10))
        
        # 스파시티 타겟 정규화
        sparsity_loss = F.mse_loss(torch.mean(importance), 1 - self.sparsity_target)
        
        return l1_loss + l0_loss + sparsity_loss
    
    def update_usage_history(self, active_slots: List[int]):
        """슬롯 사용 히스토리 업데이트"""
        usage = torch.zeros(self.num_slots)
        for slot in active_slots:
            if 0 <= slot < self.num_slots:
                usage[slot] = 1.0
        
        self.slot_usage_history = 0.9 * self.slot_usage_history + 0.1 * usage
    
    def get_pruned_slots(self) -> List[int]:
        """pruning된 슬롯 목록 반환"""
        importance = self.get_slot_importance()
        pruned = (importance < self.pruning_threshold).nonzero(as_tuple=True)[0]
        return pruned.tolist()

class MemoryAwareMixedPrecision:
    """
    Memory-Aware Mixed Precision (MAMP)
    슬롯별 gradient norm과 VRAM slack을 결합해 FP32/FP16/INT8을 동적으로 배분하는 정밀도 컨트롤러
    """
    def __init__(self,
                 precision_levels: List[str] = ['fp32', 'fp16', 'int8'],
                 memory_thresholds: List[float] = [0.7, 0.9, 1.0],
                 gradient_thresholds: List[float] = [1.0, 0.1, 0.01]):
        self.precision_levels = precision_levels
        self.memory_thresholds = memory_thresholds
        self.gradient_thresholds = gradient_thresholds
        self.slot_precision_history = {}
        
    def get_optimal_precision(self, 
                            slot_id: int,
                            gradient_norm: float,
                            memory_usage: float) -> str:
        """최적 정밀도 결정"""
        # 메모리 사용률과 gradient norm을 결합한 점수 계산
        memory_score = sum(1 for threshold in self.memory_thresholds if memory_usage < threshold)
        gradient_score = sum(1 for threshold in self.gradient_thresholds if gradient_norm > threshold)
        
        # 점수에 따른 정밀도 선택
        combined_score = memory_score + gradient_score
        precision_idx = min(combined_score, len(self.precision_levels) - 1)
        
        precision = self.precision_levels[precision_idx]
        
        # 히스토리 업데이트
        if slot_id not in self.slot_precision_history:
            self.slot_precision_history[slot_id] = []
        self.slot_precision_history[slot_id].append(precision)
        
        return precision
    
    def get_precision_context(self, precision: str):
        """정밀도별 컨텍스트 매니저"""
        if precision == 'fp32':
            return torch.no_grad()  # 실제로는 autocast(False)
        elif precision == 'fp16':
            return autocast()
        elif precision == 'int8':
            return torch.no_grad()  # INT8은 별도 구현 필요
        else:
            return torch.no_grad()

class SlotScheduler(ABC):
    """
    Slot 기반 학습 루프에서 "다음에 활성화할 슬롯 인덱스"를 반환하는 추상화 모듈
    """
    @abstractmethod
    def get_next_slots(self, epoch: int, step: int, metrics: Dict[str, float]) -> List[int]:
        """다음 활성화할 슬롯 반환"""
        pass

class RoundRobinSlotScheduler(SlotScheduler):
    """라운드 로빈 방식의 슬롯 스케줄러"""
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
    
    def get_next_slots(self, epoch: int, step: int, metrics: Dict[str, float]) -> List[int]:
        return [(epoch + step) % self.num_slots]

class AdaptiveSlotScheduler(SlotScheduler):
    """성능 기반 적응형 슬롯 스케줄러"""
    def __init__(self, num_slots: int, performance_window: int = 10):
        self.num_slots = num_slots
        self.performance_window = performance_window
        self.slot_performance = {i: [] for i in range(num_slots)}
    
    def get_next_slots(self, epoch: int, step: int, metrics: Dict[str, float]) -> List[int]:
        # 성능이 좋은 슬롯을 더 자주 선택
        if 'loss' in metrics:
            current_slot = (epoch + step) % self.num_slots
            self.slot_performance[current_slot].append(metrics['loss'])
            
            # 최근 성능 기반으로 슬롯 선택
            if len(self.slot_performance[current_slot]) >= self.performance_window:
                avg_loss = np.mean(self.slot_performance[current_slot][-self.performance_window:])
                if avg_loss < 1.0:  # 성능이 좋으면 같은 슬롯 유지
                    return [current_slot]
        
        # 기본적으로 라운드 로빈
        return [(epoch + step) % self.num_slots]

class PrecisionController:
    """
    confidence나 gradient norm 기준으로 자동으로 FP16/FP32/INT8 전환해 주는 컨텍스트 매니저
    """
    def __init__(self, 
                 confidence_threshold: float = 0.8,
                 gradient_threshold: float = 1.0,
                 memory_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.gradient_threshold = gradient_threshold
        self.memory_threshold = memory_threshold
        self.mamp = MemoryAwareMixedPrecision()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def get_precision_context(self, 
                            confidence: float = None,
                            gradient_norm: float = None,
                            memory_usage: float = None) -> Any:
        """조건에 따른 정밀도 컨텍스트 반환"""
        if memory_usage is None:
            memory_usage = self._get_memory_usage()
        
        if gradient_norm is None:
            gradient_norm = 1.0  # 기본값
        
        # 조건에 따른 정밀도 결정
        if memory_usage > self.memory_threshold:
            precision = 'fp16'
        elif confidence and confidence > self.confidence_threshold:
            precision = 'fp16'
        elif gradient_norm < self.gradient_threshold:
            precision = 'fp16'
        else:
            precision = 'fp32'
        
        return self.mamp.get_precision_context(precision)
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용률 반환"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUtil
            return 0.0
        except:
            return 0.0

class CurriculumSelfTraining:
    """
    Curriculum Self-Training (DualLoader + Pseudo-Label Curriculum)
    epoch별로 threshold를 조절하며 pseudo-label을 점진적으로 도입하는 스테이지드 학습 로직
    """
    def __init__(self,
                 confidence_thresholds: List[float] = [0.9, 0.8, 0.7, 0.6],
                 pseudo_label_ratios: List[float] = [0.1, 0.3, 0.5, 0.7],
                 warmup_epochs: int = 5):
        self.confidence_thresholds = confidence_thresholds
        self.pseudo_label_ratios = pseudo_label_ratios
        self.warmup_epochs = warmup_epochs
        self.current_stage = 0
        
    def get_current_thresholds(self, epoch: int) -> Tuple[float, float]:
        """현재 에폭에 맞는 threshold 반환"""
        if epoch < self.warmup_epochs:
            return 1.0, 0.0  # 워밍업 단계에서는 pseudo-label 사용 안함
        
        stage = min((epoch - self.warmup_epochs) // 5, len(self.confidence_thresholds) - 1)
        self.current_stage = stage
        
        confidence_threshold = self.confidence_thresholds[stage]
        pseudo_ratio = self.pseudo_label_ratios[stage]
        
        return confidence_threshold, pseudo_ratio
    
    def generate_pseudo_labels(self, 
                             model: nn.Module,
                             unlabeled_data: torch.Tensor,
                             confidence_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """pseudo-label 생성"""
        model.eval()
        with torch.no_grad():
            logits = model(unlabeled_data)
            probabilities = F.softmax(logits, dim=-1)
            confidence, predictions = torch.max(probabilities, dim=-1)
            
            # confidence threshold 적용
            mask = confidence > confidence_threshold
            pseudo_labels = predictions[mask]
            pseudo_data = unlabeled_data[mask]
            
        model.train()
        return pseudo_data, pseudo_labels
    
    def should_advance_stage(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """다음 스테이지로 진행할지 결정"""
        if epoch < self.warmup_epochs:
            return False
        
        # 성능 기준으로 스테이지 진행
        if 'loss' in metrics and metrics['loss'] < 1.0:
            return True
        
        return False

# 통합 최적화 매니저
class DORMV4AdvancedOptimizer:
    """
    DORM-V4의 모든 고급 최적화 기법을 통합 관리하는 클래스
    """
    def __init__(self, 
                 num_slots: int,
                 device: torch.device,
                 config: Dict[str, Any]):
        self.num_slots = num_slots
        self.device = device
        self.config = config
        
        # 각 최적화 기법 초기화
        self.abs = AdaptiveBatchSizing(**config.get('abs', {}))
        self.sfs = SlotFusionScheduler(num_slots, **config.get('sfs', {}))
        self.msc = MetaSlotCalibration(num_slots, **config.get('msc', {}))
        self.cbp = CurriculumBatchedPretraining(**config.get('cbp', {}))
        self.spr = SparsePriorRegularization(num_slots, **config.get('spr', {}))
        self.mamp = MemoryAwareMixedPrecision(**config.get('mamp', {}))
        self.slot_scheduler = AdaptiveSlotScheduler(num_slots, **config.get('slot_scheduler', {}))
        self.precision_controller = PrecisionController(**config.get('precision_controller', {}))
        self.cst = CurriculumSelfTraining(**config.get('cst', {}))
        
    def get_training_config(self, epoch: int, step: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """현재 상태에 맞는 최적화된 학습 설정 반환"""
        # 배치 크기 조정
        batch_size = self.abs.adjust_batch_size(metrics.get('loss'))
        
        # 커리큘럼 파라미터
        seq_len, curriculum_batch_size = self.cbp.get_curriculum_params(epoch, self.config.get('total_epochs', 100))
        
        # 슬롯 선택
        active_slots = self.slot_scheduler.get_next_slots(epoch, step, metrics)
        
        # 퓨전 슬롯 적용
        fusion_slots = self.sfs.get_fusion_slots(epoch, step)
        
        # 최종 활성 슬롯 (퓨전 + 일반)
        final_slots = list(set(active_slots + fusion_slots))
        
        return {
            'batch_size': min(batch_size, curriculum_batch_size),
            'seq_len': seq_len,
            'active_slots': final_slots,
            'fusion_slots': fusion_slots,
            'precision_context': self.precision_controller.get_precision_context(
                metrics.get('confidence'),
                metrics.get('gradient_norm'),
                self.abs.get_gpu_memory_usage()
            )
        }
    
    def update_metrics(self, slot_id: int, metrics: Dict[str, float]):
        """메트릭 업데이트 및 최적화 수행"""
        # 메타 슬롯 캘리브레이션 업데이트
        self.msc.update_performance(slot_id, metrics)
        
        # 스파시티 정규화 업데이트
        if 'active_slots' in metrics:
            self.spr.update_usage_history(metrics['active_slots'])
        
        # 슬롯 퓨전 스케줄러 업데이트
        self.sfs.update_fusion_groups(metrics)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """정규화 손실 계산"""
        return self.spr.compute_sparsity_loss()
    
    def get_slot_config(self, slot_id: int) -> SlotConfig:
        """슬롯별 설정 반환"""
        return self.msc.get_slot_config(slot_id) 
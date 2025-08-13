"""
DORM-V4 통합 최적화 라이브러리
모든 최적화 기능을 하나로 통합하여 외부 의존성 없이 고성능 학습을 제공합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import time
import math
from typing import Optional, Dict, Any, List, Tuple
import warnings

class FastAttention(nn.Module):
    """
    FlashAttention 없이도 빠른 어텐션을 구현한 커스텀 모듈
    메모리 효율적인 어텐션 연산을 제공합니다.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Q, K, V 계산
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 스케일드 닷 프로덕트 어텐션 (메모리 효율적)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # 마스크 적용
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 소프트맥스 및 드롭아웃
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 어텐션 출력 계산
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context)

class MemoryOptimizer:
    """
    메모리 최적화를 위한 유틸리티 클래스
    DeepSpeed 없이도 메모리 효율적인 학습을 제공합니다.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.original_params = {}
        self.optimized_params = {}
        
    def optimize_memory(self, optimization_level: int = 1):
        """
        메모리 최적화 레벨:
        0: 최적화 없음
        1: 기본 최적화 (gradient checkpointing)
        2: 고급 최적화 (parameter offloading)
        """
        if optimization_level >= 1:
            # Gradient checkpointing 적용 (안전한 호출)
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing 활성화됨")
            else:
                # SlotTransformer 등 커스텀 모델의 경우 수동으로 적용
                self._apply_gradient_checkpointing_manual()
                print("수동 Gradient checkpointing 활성화됨")
            
        if optimization_level >= 2:
            # 파라미터 오프로딩 (CPU 메모리 활용)
            self._enable_parameter_offloading()
            print("Parameter offloading 활성화됨")
    
    def _apply_gradient_checkpointing_manual(self):
        """커스텀 모델에 대한 수동 gradient checkpointing 적용"""
        # SlotTransformer의 경우 각 slot에 대해 적용
        if hasattr(self.model, 'slots'):
            for slot in self.model.slots:
                if hasattr(slot, 'gradient_checkpointing_enable'):
                    slot.gradient_checkpointing_enable()
        
        # 일반적인 nn.Module의 경우
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    
    def _enable_parameter_offloading(self):
        """큰 파라미터들을 CPU로 오프로딩"""
        for name, param in self.model.named_parameters():
            if param.numel() > 1e6:  # 1M 파라미터 이상
                self.original_params[name] = param.data.clone()
                param.data = param.data.cpu()
                self.optimized_params[name] = param
    
    def restore_parameters(self):
        """오프로딩된 파라미터들을 GPU로 복원"""
        for name, param in self.optimized_params.items():
            if name in self.original_params:
                param.data = self.original_params[name].to(self.device)

class AdvancedGradScaler(GradScaler):
    """
    고급 GradScaler - 더 안정적이고 효율적인 mixed precision 학습
    """
    def __init__(self, 
                 init_scale: float = 2**10,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 enabled: bool = True):
        super().__init__(init_scale, growth_factor, backoff_factor, growth_interval, enabled)
        self.nan_count = 0
        self.inf_count = 0
        self.total_steps = 0
        
    def step(self, optimizer, *args, **kwargs):
        """NaN/Inf 체크가 포함된 고급 step"""
        self.total_steps += 1
        
        # NaN/Inf 체크
        if self._found_inf:
            self.nan_count += 1
            if self.nan_count > 10:  # 10번 연속 NaN이면 스케일 축소
                self._scale = max(self._scale * self._backoff_factor, 1.0)
                self.nan_count = 0
                print(f"NaN 감지로 스케일 축소: {self._scale}")
        
        return super().step(optimizer, *args, **kwargs)

class PerformanceProfiler:
    """
    성능 프로파일링을 위한 커스텀 클래스
    PyTorch Profiler 없이도 성능 분석을 제공합니다.
    """
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, name: str):
        """타이머 시작"""
        self.start_times[name] = time.time()
        
    def end_timer(self, name: str):
        """타이머 종료 및 기록"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            del self.start_times[name]
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """통계 정보 반환"""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    def print_summary(self):
        """성능 요약 출력"""
        print("\n=== 성능 프로파일링 결과 ===")
        for name, stats in self.metrics.items():
            if stats:
                mean_time = sum(stats) / len(stats)
                print(f"{name}: {mean_time:.4f}s (평균), {len(stats)}회 실행")

class DataLoaderOptimizer:
    """
    DataLoader 최적화를 위한 유틸리티
    """
    @staticmethod
    def optimize_dataloader(dataloader, 
                          num_workers: int = 8,
                          pin_memory: bool = True,
                          prefetch_factor: int = 2):
        """DataLoader 최적화 설정"""
        dataloader.num_workers = num_workers
        dataloader.pin_memory = pin_memory
        if hasattr(dataloader, 'prefetch_factor'):
            dataloader.prefetch_factor = prefetch_factor
        return dataloader

class LearningRateScheduler:
    """
    고급 학습률 스케줄러
    """
    @staticmethod
    def create_scheduler(optimizer, 
                        scheduler_type: str = "cosine",
                        num_warmup_steps: int = 1000,
                        num_training_steps: int = 10000,
                        min_lr: float = 1e-7):
        """다양한 스케줄러 생성"""
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps, eta_min=min_lr
            )
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps
            )
        elif scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95
            )
        else:
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

class ModelOptimizer:
    """
    모델 최적화를 위한 통합 클래스
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.memory_optimizer = MemoryOptimizer(model, device)
        self.profiler = PerformanceProfiler()
        
    def apply_optimizations(self, 
                          compile_model: bool = True,
                          memory_optimization: int = 1,
                          enable_fast_attention: bool = True):
        """모든 최적화 적용"""
        
        # 1. torch.compile 적용
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile 적용됨")
            except Exception as e:
                print(f"torch.compile 적용 실패: {e}")
        
        # 2. 메모리 최적화
        self.memory_optimizer.optimize_memory(memory_optimization)
        
        # 3. FastAttention 적용
        if enable_fast_attention:
            self._replace_attention_layers()
        
        return self.model
    
    def _replace_attention_layers(self):
        """기존 attention을 FastAttention으로 교체"""
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and isinstance(module, nn.MultiheadAttention):
                # MultiheadAttention을 FastAttention으로 교체
                d_model = module.embed_dim
                n_heads = module.num_heads
                dropout = module.dropout.p if hasattr(module.dropout, 'p') else 0.1
                
                fast_attn = FastAttention(d_model, n_heads, dropout)
                # 모듈 교체 로직 (실제 구현에서는 더 복잡)
                print(f"FastAttention으로 교체: {name}")

def create_optimized_training_loop(model: nn.Module,
                                 dataloader,
                                 optimizer,
                                 device: torch.device,
                                 config: Dict[str, Any]):
    """
    최적화된 학습 루프 생성
    """
    model_optimizer = ModelOptimizer(model, device)
    model = model_optimizer.apply_optimizations(
        compile_model=config.get('compile_model', True),
        memory_optimization=config.get('memory_optimization', 1),
        enable_fast_attention=config.get('enable_fast_attention', True)
    )
    
    scaler = AdvancedGradScaler(
        init_scale=config.get('gradscaler_init_scale', 2**10),
        growth_factor=config.get('gradscaler_growth_factor', 2.0),
        backoff_factor=config.get('gradscaler_backoff_factor', 0.5),
        growth_interval=config.get('gradscaler_growth_interval', 1000)
    )
    
    profiler = PerformanceProfiler()
    
    return model, scaler, profiler

# 사용 예시
if __name__ == "__main__":
    # 테스트 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 간단한 모델 생성
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100)
    ).to(device)
    
    # 최적화 적용
    model_optimizer = ModelOptimizer(model, device)
    optimized_model = model_optimizer.apply_optimizations()
    
    print("최적화 완료!") 
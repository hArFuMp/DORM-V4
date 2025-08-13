import contextlib
import warnings
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, Any, List

# --- GPU backend (safe device index, reserved memory usage) ---
class PyTorchGpuBackend:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.total_memory = 0
        self.device_index = None
        if self.device.type != 'cpu':
            try:
                idx = getattr(self.device, "index", None)
                if idx is None:
                    idx = torch.cuda.current_device()
                self.device_index = idx
                props = torch.cuda.get_device_properties(self.device_index)
                self.total_memory = props.total_memory
            except Exception as e:
                warnings.warn(f"GPU properties unavailable: {e}")
                self.device_index = None

    def get_memory_usage(self) -> float:
        """Return GPU memory usage fraction in [0.0, 1.0]."""
        if self.device.type == 'cpu' or self.total_memory == 0 or self.device_index is None:
            return 0.0
        try:
            try:
                reserved = torch.cuda.memory_reserved(self.device_index)
            except Exception:
                reserved = torch.cuda.memory_allocated(self.device_index)
            allocated = torch.cuda.memory_allocated(self.device_index)
            used = max(allocated, reserved)
            return float(used) / float(self.total_memory)
        except Exception as e:
            warnings.warn(f"GPU memory check failed: {e}")
            return 0.0

# --- Sparse prior regularization ---
class SparsePriorRegularization(nn.Module):
    def __init__(self, num_slots: int, l1_lambda: float = 1e-4, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_slots = num_slots
        self.l1_lambda = l1_lambda
        self.slot_weights = nn.Parameter(torch.ones(num_slots, device=device))

    def compute_sparsity_loss(self) -> torch.Tensor:
        """L1 loss on sigmoid of slot weights."""
        return self.l1_lambda * torch.sum(torch.abs(torch.sigmoid(self.slot_weights)))

    def update_usage_history(self, active_slots: List[int]):
        pass  # Optional slot usage tracking

# --- Memory-aware mixed precision ---
class MemoryAwareMixedPrecision:
    def __init__(self, backend: PyTorchGpuBackend, memory_threshold: float = 0.8, device: torch.device = torch.device("cpu")):
        self.backend = backend
        self.memory_threshold = memory_threshold
        self.device = device

    def get_precision_context(self):
        """Return autocast if memory exceeds threshold, else nullcontext."""
        try:
            if self.device.type == 'cuda' and self.backend.get_memory_usage() > self.memory_threshold:
                return autocast(device_type='cuda')
        except Exception:
            pass
        return contextlib.nullcontext()

# --- Simple round-robin slot scheduler ---
class RoundRobinSlotScheduler:
    def __init__(self, num_slots: int):
        self.num_slots = num_slots

    def get_next_slots(self, epoch: int, step: int, **kwargs) -> List[int]:
        return [(epoch + step) % self.num_slots]

# --- Integrated optimizer ---
class DORMV4AdvancedOptimizer:
    def __init__(self, num_slots: int, device: torch.device, config: Dict[str, Any]):
        self.num_slots = num_slots
        self.device = device
        self.config = config or {}
        self.backend = PyTorchGpuBackend(device)

        self.spr = SparsePriorRegularization(num_slots, **config.get('spr', {}), device=device)
        self.mamp = MemoryAwareMixedPrecision(self.backend, **config.get('mamp', {}), device=device)
        self.slot_scheduler = RoundRobinSlotScheduler(num_slots)

    def get_training_config(self, epoch: int, step: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Return active slot mask and precision context."""
        indices = []
        for i in self.slot_scheduler.get_next_slots(epoch=epoch, step=step, metrics=metrics) or []:
            try:
                ii = int(i)
                if 0 <= ii < self.num_slots:
                    indices.append(ii)
            except Exception:
                continue
        indices = sorted(set(indices))

        active_mask = torch.zeros(self.num_slots, dtype=torch.bool, device=self.device)
        if indices:
            active_mask[indices] = True
        else:
            active_mask[:] = True

        return {
            'active_slots': active_mask,
            'precision_context': self.mamp.get_precision_context(),
        }

    def update_metrics(self, metrics: Dict[str, float]):
        """Update slot usage history."""
        if 'active_slots' in metrics:
            self.spr.update_usage_history(metrics['active_slots'])

    def get_regularization_loss(self) -> torch.Tensor:
        """Return current sparsity loss."""
        return self.spr.compute_sparsity_loss()

    def parameters(self):
        """Return parameters for optimizer."""
        return list(self.spr.parameters())
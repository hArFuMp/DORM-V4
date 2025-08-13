
import torch
import torch.nn.utils.prune as prune

def progressive_pruning(model, slot_idx, pruning_ratio=0.1):
    """
    특정 Slot(레이어)의 가중치를 점진적으로 제거(pruning)합니다.
    여기서는 L1-norm을 기준으로 중요도가 낮은 가중치를 제거합니다.

    Args:
        model (torch.nn.Module): 전체 SlotTransformer 모델.
        slot_idx (int): Pruning을 적용할 Slot의 인덱스.
        pruning_ratio (float): 제거할 가중치의 비율.

    Returns:
        torch.nn.Module: Pruning이 적용된 모델.
    """
    if not (0 <= slot_idx < len(model.slots)):
        print(f"Warning: Pruning slot index {slot_idx} is out of range.")
        return model

    slot_to_prune = model.slots[slot_idx]
    print(f"Applying L1 unstructured pruning to Slot {slot_idx} with ratio {pruning_ratio:.2f}")

    # Slot 내부의 Linear 레이어들을 대상으로 Pruning 수행
    for name, module in slot_to_prune.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            # Pruning을 영구적으로 적용하려면 아래 라인 활성화
            # prune.remove(module, 'weight')
            
    return model

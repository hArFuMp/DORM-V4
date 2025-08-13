
import torch

def fuse_slots(model, slot_idx1, slot_idx2, method='average'):
    """
    두 개의 Slot(레이어) 간의 가중치를 통합(fusion)합니다.

    Args:
        model (torch.nn.Module): 전체 SlotTransformer 모델.
        slot_idx1 (int): 첫 번째 Slot의 인덱스.
        slot_idx2 (int): 두 번째 Slot의 인덱스.
        method (str): 통합 방식. 'average' 또는 'soft_attention' (미구현).

    Returns:
        torch.nn.Module: 가중치 통합이 적용된 모델.
    """
    if not (0 <= slot_idx1 < len(model.slots) and 0 <= slot_idx2 < len(model.slots)):
        print(f"Warning: Fusion slot indices ({slot_idx1}, {slot_idx2}) are out of range.")
        return model

    slot1 = model.slots[slot_idx1]
    slot2 = model.slots[slot_idx2]
    print(f"Fusing weights of Slot {slot_idx1} and Slot {slot_idx2} using '{method}' method.")

    if method == 'average':
        with torch.no_grad():
            # 두 Slot의 모든 파라미터를 순회하며 평균 계산
            for param1, param2 in zip(slot1.parameters(), slot2.parameters()):
                avg_param = (param1.data + param2.data) / 2.0
                # 한 쪽 Slot(slot1)에 평균 가중치를 업데이트
                param1.data.copy_(avg_param)
    else:
        raise NotImplementedError(f"Fusion method '{method}' is not implemented.")

    return model

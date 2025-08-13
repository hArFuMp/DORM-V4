
import numpy as np

def get_curriculum_priority(num_slots, current_epoch, total_epochs):
    """
    학습 진행도에 따라 Slot의 우선순위를 결정하는 커리큘럼을 반환합니다.
    초반에는 쉬운(앞쪽) Slot에, 후반에는 어려운(뒤쪽) Slot에 집중합니다.

    Args:
        num_slots (int): 전체 Slot의 개수.
        current_epoch (int): 현재 에폭.
        total_epochs (int): 총 학습 에폭.

    Returns:
        np.array: 각 Slot에 대한 우선순위 확률분포.
    """
    # 학습 진행도를 0과 1 사이의 값으로 계산
    progress = current_epoch / total_epochs
    
    # 진행도에 따라 우선순위의 중심이 이동
    # 초반(progress=0) -> 중심=0, 후반(progress=1) -> 중심=num_slots-1
    center_slot = progress * (num_slots - 1)
    
    # 정규분포(가우시안)를 사용하여 중심 Slot에 높은 확률을 부여
    indices = np.arange(num_slots)
    # 표준편차(scale)는 고정값으로 설정하여 분포의 퍼짐 정도를 조절
    scale = num_slots / 4 
    
    # 중심에서 멀어질수록 확률이 낮아지는 가중치 계산
    probabilities = np.exp(-((indices - center_slot)**2) / (2 * scale**2))
    
    # 전체 확률의 합이 1이 되도록 정규화
    probabilities /= probabilities.sum()
    
    return probabilities

def choose_slot_by_curriculum(num_slots, current_epoch, total_epochs):
    """
    커리큘럼 우선순위에 따라 학습할 Slot을 하나 선택합니다.

    Args:
        num_slots (int): 전체 Slot의 개수.
        current_epoch (int): 현재 에폭.
        total_epochs (int): 총 학습 에폭.

    Returns:
        int: 선택된 Slot의 인덱스.
    """
    probabilities = get_curriculum_priority(num_slots, current_epoch, total_epochs)
    chosen_slot = np.random.choice(np.arange(num_slots), p=probabilities)
    return chosen_slot

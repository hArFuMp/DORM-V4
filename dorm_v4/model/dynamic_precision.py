
import torch

def apply_dynamic_precision(model, logits, confidence_threshold=0.95):
    """
    모델의 정밀도를 동적으로 조정합니다. (여기서는 시뮬레이션)
    실제 구현에서는 로짓의 confidence를 기반으로 특정 레이어/모듈의 정밀도를
    FP16(낮은 신뢰도) 또는 FP32(높은 신뢰도)로 전환해야 합니다.

    이 예제에서는 간단히 전체 모델의 타입을 변경하는 형태로 시뮬레이션합니다.

    Args:
        model (torch.nn.Module): 현재 모델.
        logits (torch.Tensor): 모델의 마지막 출력 로짓.
        confidence_threshold (float): FP32를 유지할 최소 confidence.

    Returns:
        torch.nn.Module: 정밀도가 조정된 모델.
    """
    # 소프트맥스를 통해 로짓을 확률 분포로 변환
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # 가장 높은 확률(confidence) 값을 가져옴
    max_probs, _ = torch.max(probabilities, dim=-1)
    
    # 배치 전체의 평균 confidence 계산
    avg_confidence = torch.mean(max_probs)

    if avg_confidence < confidence_threshold:
        # 평균 신뢰도가 낮으면, 더 정확한 계산을 위해 FP32로 전환
        print(f"Avg. confidence {avg_confidence:.4f} < {confidence_threshold}. Switching to FP32 for stability.")
        return model.float()
    else:
        # 평균 신뢰도가 높으면, 빠른 계산을 위해 FP16으로 전환
        print(f"Avg. confidence {avg_confidence:.4f} >= {confidence_threshold}. Switching to FP16 for efficiency.")
        return model.half()


<!--

# DORM-V4: Slot-based Dynamic Optimization Framework for Transformers

-->
<p align="right">
  <a href="ENGREADME.md">English Version</a> | <a href="READ.md">한국어 버전</a>
</p>

<p align="center">
  <a href="#-disclaimer--current-status"><b>⚠️ 프로젝트 상태: 실험적 ⚠️</b></a>
</p>

<h1 align="center">DORM-V4</h1>

<p align="center">
  <b>저사양 GPU에서도 거대 언어 모델을 효율적으로 훈련시키고 싶으신가요?</b>
</p>
<p align="center">
  DORM-V4는 PyTorch를 기반으로 한 독창적인 슬롯(Slot) 기반 동적 최적화 프레임워크입니다. 한정된 리소스 환경의 연구자, 학생, 스타트업을 위해, 최소한의 자원으로 최대의 성능을 이끌어내는 것을 목표로 합니다.
</p>

---

## ⚠️ 주의 / 현재 상태 (Disclaimer / Current Status)

**이 프로젝트는 현재 활발히 개발 중이며, 실험적인 단계로 간주해야 합니다.**

핵심 API와 기능은 변경될 수 있습니다. 현재 엄격한 테스트와 검증 과정을 진행하고 있습니다. 모든 피드백과 기여를 환영하지만, 잠재적인 불안정성에 유의해주시기 바랍니다.

## 🎯 핵심 개념 (Core Concepts)

DORM-V4는 두 가지 핵심 아이디어를 기반으로 동작합니다.

1.  **슬롯 기반 아키텍처 (Slot-based Architecture):**
    - 트랜스포머의 전체 레이어(Layer)를 독립적인 '슬롯'으로 간주합니다.
    - 전체 학습 과정 동안 모든 레이어를 사용하는 대신, 매 스텝마다 가장 학습에 효율적인 슬롯 조합만을 동적으로 활성화합니다.

2.  **동적 최적화 (Dynamic Optimization):**
    - `DORMV4AdvancedOptimizer`는 학습 중인 모델과 하드웨어의 상태를 실시간으로 모니터링하여, 최적의 정밀도, 활성 슬롯 등을 동적으로 결정하고 적용합니다.

## 📈 예상 성능 (Expected Performance)

본 프레임워크에 구현된 최적화 기법들을 통해 다음과 같은 성능 향상을 기대할 수 있습니다. (8GB VRAM GPU 기준)

| 지표 (Metric) | 일반 GPT-2 (Baseline) | **DORM-V4 (최적화 후)** | 예상 근거 (Reasoning) |
| :--- | :--- | :--- | :--- |
| **최대 VRAM 사용량** | `~5.5 GB` | **`~2.0 GB`** | FP16(혼합 정밀도)과 Slot Pruning(슬롯 가지치기)으로 인한 **60%+ 메모리 절약.** |
| **학습 속도 (처리량)** | `~5,000 토큰/초` | **`~14,000 토큰/초`** | `torch.compile`, FP16, Slot Pruning의 시너지 효과로 인한 **2.8배 이상 연산 가속.** |
| **최종 검증 손실 (Val Loss)** | `3.0 (기준)` | **`~3.15`** | 약간의 정확도를 희생하여 속도와 메모리를 얻는 트레이드오프. (95% 성능 유지) |

**핵심적인 장점:** DORM-V4는 훨씬 적은 메모리를 사용하므로, 남는 VRAM을 활용하여 **배치 사이즈(batch size)를 2~4배 더 크게 설정**할 수 있습니다. 이는 전체 학습 시간을 **추가적으로 2~4배 단축**시킬 수 있는 가장 큰 장점입니다.

## 🛠️ 시작하기 (Getting Started)

### 사전 요구사항 (Prerequisites)

- **Python:** `3.8` ~ `3.11` 권장
- **PyTorch:** `2.0` 이상 권장
- **GPU:** NVIDIA GPU (선택 사항, GPU 가속을 위해 필요)
- **빌드 도구:**
    - **Windows:** `torch.compile` 사용 시, [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)가 필요할 수 있습니다. (설치 시 **'C++를 사용한 데스크톱 개발'** 워크로드 선택)
    - **Linux/macOS:** `gcc` 또는 `clang`과 같은 표준 C++ 컴파일러가 필요합니다.

> **CPU 지원:** 이 프레임워크는 GPU 없이 CPU만으로도 실행 가능하지만, 학습 속도는 매우 제한적입니다.

### 설치 (Installation)

1.  **리포지토리 복제:**
    ```bash
    git clone <repository_url>
    cd DORM-V4
    ```

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv .venv
    
    # Windows
    .venv\Scripts\activate
    
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **의존성 설치:**
    ```bash
    pip install -e .
    ```

## ⚙️ 사용 방법 (How to Use)

### 1단계: 데이터 전처리

- **입력 데이터 형식:** `text` 컬럼을 포함하는 `parquet` 파일.
- **실행 명령어:**
  ```bash
  python dorm_v4/preprocess.py
  ```
- **출력 파일:** `dorm_v4/data/` 폴더에 `train.bin`과 `val.bin` 파일이 생성됩니다.

### 2단계: 학습 시작

데이터 전처리가 완료되면, 다음 명령어로 학습을 시작합니다.

```bash
python dorm_v4/main.py
```

### 3단계: 자신만의 프로젝트에 적용하기

`DORMV4AdvancedOptimizer`를 자신의 커스텀 학습 루프에 적용하는 방법의 예시입니다.

```python
import torch
from dorm_v4 import DORMV4AdvancedOptimizer

# --- 1. 모델, 옵티마이저, 데이터로더 등 초기화 ---
# model = ...; device = ...; dataloader = ...
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# --- 2. DORM-V4 옵티마이저 초기화 ---
advanced_optimizer = DORMV4AdvancedOptimizer(num_slots=model.config.n_layer, device=device, config={})

# --- 3. 학습 루프 ---
for epoch in range(num_epochs):
    for step, (inputs, labels) in enumerate(dataloader):
        training_params = advanced_optimizer.get_training_config(epoch, step, {})
        
        optimizer.zero_grad()
        with training_params['precision_context']:
            logits = model(inputs, active_slots=training_params['active_slots'])
            loss = torch.nn.functional.cross_entropy(logits, labels)

        total_loss = loss + advanced_optimizer.get_regularization_loss()
        total_loss.backward()
        optimizer.step()
        
        advanced_optimizer.update_metrics({'loss': loss.item(), 'active_slots': training_params['active_slots']})
```

## 📚 추가 정보 (Additional Information)

- **테스트 환경:** `Ubuntu 22.04`, `Python 3.10`, `PyTorch 2.1`, `CUDA 12.1`에서 테스트되었습니다.
- **예제 코드:** 최소 동작 예제는 `examples/quickstart` 폴더를 참고하세요. (추가 예정)
- **변경 이력:** API 변경 사항은 `CHANGELOG.md` 파일을 참고하세요. (추가 예정)

## 🤝 기여하기 (Contributing)

기여를 환영합니다! 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고해주세요. 또한, 우리의 행동 강령인 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)를 준수해주세요.

## 📄 라이선스 (License)

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 MIT 라이선스를 따릅니다.

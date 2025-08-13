import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class SlotTransformer(nn.Module):
    """
    Slot 기반의 동적 Transformer 모델.
    전체 레이어 중 선택된 'Slot'(레이어)만 활성화하여 연산을 수행합니다.
    """
    def __init__(self, config):
        """
        Args:
            config: 모델의 설정을 담은 객체 (예: Hugging Face의 GPT2Config).
                    config 객체에는 num_layers, d_model, n_head 등의 정보가 포함되어야 합니다.
        """
        super().__init__()
        self.config = config
        self.num_slots = config.num_hidden_layers
        self.gradient_checkpointing = False
        
        # 전체 Transformer 레이어(Slot)를 리스트로 정의
        self.slots = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(self.num_slots)])
        
        # 단어 임베딩 및 위치 임베딩
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # 최종 Layer Normalization 및 언어 모델 헤드
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def gradient_checkpointing_enable(self):
        """Gradient checkpointing을 활성화합니다."""
        self.gradient_checkpointing = True
        # 각 slot에 대해서도 gradient checkpointing 적용
        for slot in self.slots:
            if hasattr(slot, 'gradient_checkpointing_enable'):
                slot.gradient_checkpointing_enable()
        print("SlotTransformer: Gradient checkpointing 활성화됨")

    def gradient_checkpointing_disable(self):
        """Gradient checkpointing을 비활성화합니다."""
        self.gradient_checkpointing = False
        # 각 slot에 대해서도 gradient checkpointing 비활성화
        for slot in self.slots:
            if hasattr(slot, 'gradient_checkpointing_disable'):
                slot.gradient_checkpointing_disable()
        print("SlotTransformer: Gradient checkpointing 비활성화됨")

    def forward(self, input_ids, attention_mask, active_slots):
        """
        모델의 순전파를 수행합니다.

        Args:
            input_ids (torch.Tensor): 입력 토큰 ID (batch_size, seq_len).
            attention_mask (torch.Tensor): 어텐션 마스크 (batch_size, seq_len).
            active_slots (torch.Tensor[bool]): 이번 forward pass에서 활성화할 slot을 나타내는 boolean 마스크 텐서.

        Returns:
            torch.Tensor: 모델의 출력 로짓 (batch_size, seq_len, vocab_size).
        """
        # 1. 입력 임베딩 생성
        inputs_embeds = self.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # attention_mask가 2D면 4D로 변환 (batch, 1, 1, seq_len)
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        # 2. 선택된 active_slots에 대해서만 Transformer Block 연산 수행
        # Python list 대신 Tensor 마스크를 사용하여 JIT 컴파일러에 친화적으로 변경
        for i, layer in enumerate(self.slots):
            if active_slots[i]:
                if self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask=attention_mask)[0]
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask)
                    hidden_states = layer_outputs[0]

        # 3. 최종 출력 계산
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def resize_slots(self, new_num_slots):
        """
        Adaptive Slot Resizing (ASR)을 수행합니다.
        모델의 전체 slot 개수를 동적으로 조절합니다.
        
        Args:
            new_num_slots (int): 새로 설정할 slot의 개수.
        """
        if new_num_slots == self.num_slots:
            return

        if new_num_slots > self.num_slots:
            # Slot 추가
            for i in range(self.num_slots, new_num_slots):
                self.slots.append(GPT2Block(self.config, layer_idx=i).to(self.wte.weight.device))
        else:
            # Slot 제거
            self.slots = nn.ModuleList(list(self.slots)[:new_num_slots])
        
        self.num_slots = new_num_slots
        print(f"Model slots resized to {new_num_slots}")

if __name__ == '__main__':
    from transformers import GPT2Config

    # --- 사용 예시 ---
    # 1. 모델 설정 정의 (GPT-2 small 기준)
    config = GPT2Config(
        vocab_size=51200,      # 예시 vocab size
        n_positions=1024,      # 최대 시퀀스 길이
        n_embd=768,            # 임베딩 차원
        n_layer=12,            # 전체 레이어(slot) 수
        n_head=12,             # 어텐션 헤드 수
        n_inner=3072,          # Feed-forward 네트워크 내부 차원
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )

    # 2. SlotTransformer 모델 생성
    model = SlotTransformer(config)
    model.eval() # 평가 모드로 설정

    # 3. 더미 입력 데이터 생성
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 4. 학습할 Slot 동적 선택 (예: 0, 2, 5, 10번 레이어만 사용)
    active_slots = [0, 2, 5, 10]

    # 5. 모델 Forward Pass 실행
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask, active_slots=active_slots)
    
    print(f"Running with {len(active_slots)} active slots: {active_slots}")
    print("Output logits shape:", logits.shape)

    # 6. Adaptive Slot Resizing 테스트
    print(f"\nOriginal number of slots: {model.num_slots}")
    model.resize_slots(16) # 16개로 늘리기
    print(f"Resized number of slots: {model.num_slots}")
    model.resize_slots(8)  # 8개로 줄이기
    print(f"Resized number of slots: {model.num_slots}")

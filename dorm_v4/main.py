import torch
import wandb
from tqdm import tqdm
from transformers import GPT2Config, PreTrainedTokenizerFast
from torch.optim import AdamW
import time
import os

# DORM-V4 통합 최적화 라이브러리
from utils.optimizer import (
    ModelOptimizer, 
    AdvancedGradScaler, 
    PerformanceProfiler,
    LearningRateScheduler,
)

from config import ProjectConfig
from data.dataset import get_dataloader
from model.slot_transformer import SlotTransformer
from utils.logger import setup_wandb
from dorm_v4 import DORMV4AdvancedOptimizer

def main():
    """ DORM-V4의 전체 학습 파이프라인을 실행합니다. """
    
    # 1. 설정 로드
    config = ProjectConfig()
    model_config_dict = vars(config.model)
    data_config = config.data
    train_config = config.training

    # 2. 로깅 설정 (Wandb)
    wandb_run = setup_wandb(train_config, config.model)

    # 3. 토크나이저 로드 (vocab_size 설정용)
    # 데이터 로딩 자체는 preprocess.py에서 이미 완료됨
    tokenizer_path = os.path.join("DORM-V4", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"오류: 토크나이저 파일({tokenizer_path})을 찾을 수 없습니다.")
        return
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model_config_dict['vocab_size'] = tokenizer.vocab_size
    model_config = GPT2Config(**model_config_dict)

    # 4. 데이터로더 준비 (미리 토큰화된 .bin 파일 사용)
    try:
        train_dataloader, eval_dataloader = get_dataloader(data_config, use_pretokenized=True)
        print("미리 토큰화된 데이터로더가 준비되었습니다.")
    except FileNotFoundError as e:
        print(f"데이터 로딩 중 오류 발생: {e}")
        print("먼저 `python DORM-V4/dorm_v4/preprocess.py`를 실행하세요.")
        return

    # 5. 모델, 옵티마이저, 및 고급 최적화 매니저 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    model = SlotTransformer(model_config).to(device)
    
    model_optimizer = ModelOptimizer(model, device)
    model = model_optimizer.apply_optimizations(
        compile_model=True,
        memory_optimization=False, # Gradient Checkpointing
        enable_fast_attention=True
    )

    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    
    num_training_steps = len(train_dataloader) * train_config.num_train_epochs
    lr_scheduler = LearningRateScheduler.create_scheduler(
        optimizer, train_config.lr_scheduler_type, train_config.lr_warmup_steps, num_training_steps
    )
    
    scaler = AdvancedGradScaler(enabled=(device.type == 'cuda'))
    profiler = PerformanceProfiler()

    advanced_optimizer_config = {
        'spr': {'l1_lambda': 1e-5},
        'mamp': {'memory_threshold': 0.8}
    }
    advanced_optimizer = DORMV4AdvancedOptimizer(model_config.n_layer, device, advanced_optimizer_config)
    
    print("DORM-V4 Advanced Optimizer가 성공적으로 초기화되었습니다.")

    # 6. 학습 루프 시작
    print("\n--- 학습을 시작합니다 ---")
    global_step = 0
    for epoch in range(train_config.num_train_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_config.num_train_epochs}")
        accumulated_metrics = {}

        for step, (input_ids, labels) in enumerate(progress_bar):
            profiler.start_timer("training_step")
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            training_params = advanced_optimizer.get_training_config(epoch, global_step, accumulated_metrics)
            active_slots = training_params['active_slots']
            precision_context = training_params['precision_context']

            with precision_context:
                logits = model(input_ids, attention_mask=None, active_slots=active_slots)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, model_config.vocab_size), labels.view(-1))

            reg_loss = advanced_optimizer.get_regularization_loss().to(device)
            total_loss = loss + reg_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1

            # 메트릭 누적
            accumulated_metrics['loss'] = accumulated_metrics.get('loss', 0) + loss.item()
            
            if (step + 1) % train_config.logging_interval == 0:
                avg_loss = accumulated_metrics['loss'] / train_config.logging_interval
                log_data = {'train_loss': avg_loss, 'epoch': epoch + 1, 'step': global_step}
                if wandb_run: wandb_run.log(log_data)
                progress_bar.set_postfix(loss=f"{avg_loss:.3f}")
                accumulated_metrics = {}

            profiler.end_timer("training_step")

        # --- Epoch 종료 후 평가 ---
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for (input_ids, labels) in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                # 평가 시에는 모든 슬롯을 활성화
                all_slots = torch.ones(model_config.n_layer, dtype=torch.bool, device=device)
                logits = model(input_ids, attention_mask=None, active_slots=all_slots)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, model_config.vocab_size), labels.view(-1))
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1} | Validation Loss: {avg_eval_loss:.4f}")
        if wandb_run: wandb_run.log({"eval_loss": avg_eval_loss, "epoch": epoch + 1})

    print("--- 학습이 완료되었습니다 ---")
    profiler.print_summary()

if __name__ == '__main__':
    main()
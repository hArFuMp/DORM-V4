import os
import time
import contextlib
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast
import torch.nn as nn

import wandb
from tqdm import tqdm
from transformers import GPT2Config, PreTrainedTokenizerFast

# DORM-V4 Optimization Library
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
    """Run DORM-V4 training pipeline."""

    # 1. Load config
    config = ProjectConfig()
    model_cfg_dict = vars(config.model)
    data_cfg = config.data
    train_cfg = config.training

    # 2. Setup wandb (safe)
    try:
        wandb_run = setup_wandb(train_cfg, config.model)
    except Exception as e:
        print(f"[WARN] wandb setup failed: {e}")
        wandb_run = None

    # 3. Load tokenizer from repo root
    repo_root = Path(__file__).resolve().parent.parent
    tokenizer_path = repo_root / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: tokenizer file not found at {tokenizer_path}")
        return
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    model_cfg_dict["vocab_size"] = tokenizer.vocab_size
    model_config = GPT2Config(**model_cfg_dict)

    # 4. Prepare dataloaders
    try:
        train_loader, eval_loader = get_dataloader(data_cfg, use_pretokenized=True)
        # quick sanity check
        try:
            sample = next(iter(train_loader))
            assert isinstance(sample, (list, tuple)) and len(sample) >= 2
        except Exception as e:
            print(f"[WARN] dataloader sample check failed: {e}")
        print("Pre-tokenized dataloaders prepared.")
    except FileNotFoundError as e:
        print(f"Error during data loading: {e}")
        print("Please run `python DORM-V4/dorm_v4/preprocess.py` first.")
        return

    # 5. Device and model init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    model = SlotTransformer(model_config).to(device)

    # 6. Model optimizations with safe fallback
    model_optimizer = ModelOptimizer(model, device)
    try:
        model = model_optimizer.apply_optimizations(
            compile_model=True,
            memory_optimization=False,  # gradient checkpointing
            enable_fast_attention=True,
        )
    except Exception as e:
        print(f"[WARN] apply_optimizations failed, using base model: {e}")

    # 7. Advanced optimizer manager
    advanced_optimizer_cfg = {
        "spr": {"l1_lambda": 1e-5},
        "mamp": {"memory_threshold": 0.8},
    }
    advanced_optimizer = DORMV4AdvancedOptimizer(model_config.n_layer, device, advanced_optimizer_cfg)
    print("DORM-V4 Advanced Optimizer initialized successfully.")

    # 8. Build optimizer (include extra params if provided)
    try:
        extra_params = advanced_optimizer.parameters()
    except Exception:
        extra_params = []
    optimizer = AdamW(list(model.parameters()) + list(extra_params),
                      lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    # 9. LR scheduler, scaler, profiler
    num_training_steps = len(train_loader) * train_cfg.num_train_epochs
    lr_scheduler = LearningRateScheduler.create_scheduler(
        optimizer, train_cfg.lr_scheduler_type, train_cfg.lr_warmup_steps, num_training_steps
    )
    scaler = AdvancedGradScaler(enabled=(device.type == "cuda"))
    profiler = PerformanceProfiler()

    # 10. Training loop
    print("\n--- Starting Training ---")
    global_step = 0

    for epoch in range(train_cfg.num_train_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_train_epochs}")
        accumulated_loss = 0.0
        local_step = 0

        for step, (input_ids, labels) in enumerate(progress_bar):
            profiler.start_timer("training_step")
            try:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # get training config from advanced optimizer
                training_params = advanced_optimizer.get_training_config(epoch, global_step, {"loss": accumulated_loss})
                active_slots = training_params.get("active_slots", None)
                precision_context = training_params.get("precision_context", None) or contextlib.nullcontext()

                # normalize active_slots to mask tensor when possible
                active_for_model = None
                if isinstance(active_slots, torch.Tensor):
                    # if boolean mask, use as-is
                    if active_slots.dtype == torch.bool:
                        active_for_model = active_slots.to(device)
                    else:
                        # numeric tensor -> convert to list of ints
                        try:
                            active_for_model = [int(x) for x in active_slots.tolist()]
                        except Exception:
                            active_for_model = None
                elif isinstance(active_slots, (list, tuple)):
                    active_for_model = list(active_slots)
                else:
                    active_for_model = None  # model should fallback to all-active

                # forward/backward under precision context
                with precision_context:
                    logits = None
                    try:
                        # attempt mask/list as provided
                        logits = model(input_ids, attention_mask=None, active_slots=active_for_model)
                    except Exception:
                        # if model rejects mask type, try converting mask->indices
                        if isinstance(active_for_model, torch.Tensor) and active_for_model.dtype == torch.bool:
                            idxs = active_for_model.nonzero(as_tuple=False).squeeze(-1).tolist()
                            logits = model(input_ids, attention_mask=None, active_slots=idxs)
                        else:
                            # final fallback: call without active_slots
                            logits = model(input_ids, attention_mask=None)

                    loss = torch.nn.functional.cross_entropy(logits.view(-1, model_config.vocab_size), labels.view(-1))

                # regularization loss (ensure tensor on device)
                reg_loss_raw = advanced_optimizer.get_regularization_loss()
                if isinstance(reg_loss_raw, torch.Tensor):
                    reg_loss = reg_loss_raw.to(device)
                else:
                    reg_loss = torch.as_tensor(reg_loss_raw, device=device)

                total_loss = loss + reg_loss

                # backward with scaler, unscale, clip, step
                scaler.scale(total_loss).backward()
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    # some scalers may not implement unscale_
                    pass

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                # zero grads and step lr
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                # metrics accumulation & logging
                accumulated_loss += float(loss.item())
                local_step += 1
                if (step + 1) % train_cfg.logging_interval == 0:
                    avg_loss = accumulated_loss / max(1, local_step)
                    log_data = {"train_loss": avg_loss, "epoch": epoch + 1, "step": global_step}
                    if wandb_run:
                        try:
                            wandb_run.log(log_data)
                        except Exception:
                            pass
                    progress_bar.set_postfix(loss=f"{avg_loss:.3f}")
                    accumulated_loss = 0.0
                    local_step = 0

            except Exception as err:
                # log and continue training loop
                print(f"[ERROR] training step failed: {err}")
            finally:
                profiler.end_timer("training_step")

        # 11. Evaluation per epoch
        model.eval()
        eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for (input_ids, labels) in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}"):
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # all slots active mask
                all_slots_mask = torch.ones(model_config.n_layer, dtype=torch.bool, device=device)
                try:
                    logits = model(input_ids, attention_mask=None, active_slots=all_slots_mask)
                except Exception:
                    idxs = all_slots_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                    logits = model(input_ids, attention_mask=None, active_slots=idxs)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, model_config.vocab_size), labels.view(-1))
                eval_loss += float(loss.item())
                eval_steps += 1

        avg_eval_loss = eval_loss / max(1, eval_steps)
        print(f"Epoch {epoch+1} | Validation Loss: {avg_eval_loss:.4f}")
        if wandb_run:
            try:
                wandb_run.log({"eval_loss": avg_eval_loss, "epoch": epoch + 1})
            except Exception:
                pass

    print("--- Training Complete ---")
    profiler.print_summary()


if __name__ == "__main__":
    main()
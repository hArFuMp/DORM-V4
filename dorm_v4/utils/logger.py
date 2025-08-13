
import wandb

def setup_wandb(training_config, model_config):
    """
    Weights & Biases 로깅을 설정하고 초기화합니다.

    Args:
        training_config: 학습 관련 설정을 담은 객체.
        model_config: 모델 아키텍처 설정을 담은 객체.

    Returns:
        wandb.run: 초기화된 wandb 실행 객체.
    """
    try:
        run = wandb.init(
            project=training_config.wandb_project,
            name=training_config.wandb_run_name,
            config={
                "training": training_config,
                "model": model_config
            }
        )
        print(f"Wandb run '{training_config.wandb_run_name}' started successfully.")
        return run
    except Exception as e:
        print(f"Could not initialize Wandb: {e}")
        print("Proceeding without Wandb logging.")
        return None

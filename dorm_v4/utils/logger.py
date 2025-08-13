import wandb

def setup_wandb(training_config, model_config):
    """
    Sets up and initializes Weights & Biases logging.

    Args:
        training_config: Object containing training configurations.
        model_config: Object containing model architecture configurations.

    Returns:
        wandb.run: Initialized wandb run object.
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
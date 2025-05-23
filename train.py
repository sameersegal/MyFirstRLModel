from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train[:1%]")

def reward_fn(completions, **_):
    return [-abs(20-len(c)) for c in completions]

cfg = GRPOConfig(
    output_dir       = "mistral-grpo-demo",
    num_generations = 2,
    per_device_train_batch_size = 2,       
    num_train_epochs = 1,
    logging_steps    = 10,
    gradient_accumulation_steps = 1,
    fp16            = True,  
    report_to="wandb",
)

trainer = GRPOTrainer(
    model        = "mistralai/Mistral-7B-Instruct-v0.3",
    args         = cfg,
    reward_funcs = reward_fn,
    train_dataset = dataset,
)


def main():
    from accelerate import Accelerator
    accelerator = Accelerator()

    try:
        trainer.train()
        # save the model
        trainer.save_model()
    finally:
        accelerator.end_training()   # clean shutdown on every rank

if __name__ == "__main__":
    
    main()
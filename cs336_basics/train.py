import os
from argparse import ArgumentParser
import numpy as np
import torch
from cs336_basics.TransFormerLM import TransformerLM
from cs336_basics.config import load_config, FullConfig
from cs336_basics.optimizers.AdamW import AdamW
from cs336_basics.transformer_layers.utils import (
    SequenceDataset,
    cross_entropy,
    gradient_clipping,
    learning_rate_scheduler,
    load_checkpoint,
    save_checkpoint,
)


def main(config: FullConfig):
    """
    Main training loop for the TransformerLM.
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta,
    ).to(device)

    # Load data
    if not os.path.exists(config.data.train_data_path):
        raise FileNotFoundError(f"Training data file not found: {config.data.train_data_path}")
    if not os.path.exists(config.data.val_data_path):
        raise FileNotFoundError(f"Validation data file not found: {config.data.val_data_path}")

    train_raw_data = np.memmap(config.data.train_data_path, dtype=np.uint16, mode="r")
    val_raw_data = np.memmap(config.data.val_data_path, dtype=np.uint16, mode="r")

    train_dataset = SequenceDataset(train_raw_data, context_length=config.model.context_length)
    val_dataset = SequenceDataset(val_raw_data, context_length=config.model.context_length)

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = device == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=(config.training.optimizer.beta1, config.training.optimizer.beta2),
        weight_decay=config.training.optimizer.weight_decay,
        eps=config.training.optimizer.eps,
    )

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_iterations = config.training.total_epochs * steps_per_epoch
    scheduler = lambda it: learning_rate_scheduler(
        it,
        config.training.lr,
        config.training.min_lr,
        config.training.warmup_steps,
        total_iterations,
    )

    # Check for checkpoint
    iteration = 0
    if os.path.exists(config.training.checkpoint_path):
        checkpoint_files = [
            f for f in os.listdir(config.training.checkpoint_path) if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        if checkpoint_files:
            latest_checkpoint_file = max(checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            checkpoint_file_path = os.path.join(config.training.checkpoint_path, latest_checkpoint_file)
            print(f"Loading checkpoint from {checkpoint_file_path}")
            iteration = load_checkpoint(checkpoint_file_path, model, optimizer)

    start_epoch = iteration // steps_per_epoch

    # Training loop
    model.train()
    for epoch in range(start_epoch, config.training.total_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            loss = cross_entropy(logits.view(-1, config.model.vocab_size), y.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=config.training.max_l2_norm)
            
            # Update learning rate
            lr = scheduler(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            optimizer.step()

            # Logging and evaluation
            if iteration % config.training.log_interval == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.4f}, Learning Rate: {lr:.6f}")

            if iteration > 0 and iteration % config.training.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        logits_val = model(x_val)
                        val_loss += cross_entropy(
                            logits_val.view(-1, config.model.vocab_size), y_val.view(-1)
                        ).item()
                    val_loss /= len(val_loader)
                    print(f"Iteration {iteration}, Validation Loss: {val_loss:.4f}")
                model.train()

            # Save checkpoint
            if iteration > 0 and iteration % config.training.save_interval == 0:
                os.makedirs(config.training.checkpoint_path, exist_ok=True)
                save_path = os.path.join(config.training.checkpoint_path, f"checkpoint_{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, save_path)
                print(f"Checkpoint saved at iteration {iteration}")

            iteration += 1

        print(f"Epoch {epoch + 1}/{config.training.total_epochs} completed.")

    # Save final checkpoint
    os.makedirs(config.training.checkpoint_path, exist_ok=True)
    final_checkpoint_path = os.path.join(config.training.checkpoint_path, f"checkpoint_{iteration}.pt")
    save_checkpoint(model, optimizer, iteration, final_checkpoint_path)
    print(f"Final model checkpoint saved at {final_checkpoint_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run TransformerLM with a specified configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
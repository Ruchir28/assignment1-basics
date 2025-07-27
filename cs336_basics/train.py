from argparse import ArgumentParser
from cs336_basics.TransFormerLM import TransformerLM
from cs336_basics.transformer_layers.utils import SequenceDataset, cross_entropy, load_checkpoint, gradient_clipping, learning_rate_scheduler, save_checkpoint
from cs336_basics.optimizers.AdamW import AdamW
import numpy as np
import torch
import os


def main():
    parser = ArgumentParser(description="Run TransformerLM with specified parameters.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for input sequences")
    parser.add_argument("--train_data_path", type=str, help="Path to training numpy file with tokenized sequences")
    parser.add_argument("--val_data_path", type=str, help="Path to validation numpy file with tokenized sequences")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint",help="Path to model checkpoint directory") 
    parser.add_argument("--total_epochs", type=int, default=10, help="Total number of training epochs")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="Maximum L2 norm for gradient clipping")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the optimizer")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps for learning rate scheduling")
    parser.add_argument("--eval_interval", type=int, default=2, help="Interval for evaluation during training")
    parser.add_argument("--save_interval", type=int, default=2, help="Interval for saving checkpoints during training")
    
    
    args = parser.parse_args()
    
    

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    )
    
    train_raw_data = np.memmap(args.train_data_path, dtype=np.int64, mode='r')
    val_raw_data = np.memmap(args.val_data_path, dtype=np.int64, mode='r')
    
    dataset = SequenceDataset(train_raw_data, context_length=args.context_length)
    val_dataset = SequenceDataset(val_raw_data, context_length=args.context_length)
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )


    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=False
                    )
    
    steps_per_epoch = len(train_loader)
    total_iterations = args.total_epochs * steps_per_epoch

    scheduler = lambda it: learning_rate_scheduler(it, args.lr, args.min_lr, args.warmup_steps, total_iterations)

    
    if os.path.exists(args.checkpoint_path):
        # check for latest checkpoint
        checkpoint_files = [f for f in os.listdir(args.checkpoint_path) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(args.checkpoint_path, latest_checkpoint)
            print(f"Loading checkpoint from {checkpoint_path}")
            iteration = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            print("No checkpoint found, starting from scratch.")
            iteration = 0
    else:
        iteration = 0


    start_epoch = iteration // steps_per_epoch

    model.train()

    for epoch in range(start_epoch, args.total_epochs):

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=args.max_l2_norm)
            optimizer.step()
        
            lr = scheduler(iteration)
        
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iteration % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        logits = model(x_val)
                        val_loss += cross_entropy(logits.view(-1, args.vocab_size), y_val.view(-1)).item()
                    val_loss /= len(val_loader)
                    print(f"Iteration {iteration}, Validation Loss: {val_loss:.4f}")
                model.train()

            if iteration % args.save_interval == 0:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                save_path = os.path.join(args.checkpoint_path, f"checkpoint_{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, save_path)
                print(f"Checkpoint saved at iteration {iteration}")

            print(f"Iteration {iteration}, Loss: {loss.item():.4f}, Learning Rate: {lr:.6f}")

            iteration += 1
        
        print(f"Epoch {epoch + 1}/{args.total_epochs} completed.")
        
    
    os.makedirs(args.checkpoint_path, exist_ok=True)
    final_checkpoint_path = os.path.join(args.checkpoint_path, f"checkpoint_{total_iterations}.pt")
    save_checkpoint(model, optimizer, iteration, final_checkpoint_path)
    print(f"Final model checkpoint saved at {final_checkpoint_path}")


if __name__ == "__main__":
    main()
import os
import torch
import argparse
import datetime
from torch.utils.data import DataLoader, random_split
from datasets_def import TextOnlyDataset, DINOCaptionDataset
from model import DualStreamTransformer
from trainer import Trainer
from transformers import BertTokenizerFast
from utils import load_and_concatenate_dino_data, load_and_concatenate_text_only_data

def create_dataloaders(tokenizer, text_only_data, dino_embeddings, captions, batch_size=32, num_workers=4, seed=42):
    text_dataset = TextOnlyDataset(text_only_data, tokenizer)
    image_dataset = DINOCaptionDataset(dino_embeddings, captions, tokenizer)

    train_split, val_split = 0.8, 0.1
    text_train_size = int(len(text_dataset) * train_split)
    text_val_size = int(len(text_dataset) * val_split)
    text_test_size = len(text_dataset) - text_train_size - text_val_size

    image_train_size = int(len(image_dataset) * train_split)
    image_val_size = int(len(image_dataset) * val_split)
    image_test_size = len(image_dataset) - image_train_size - image_val_size

    text_train, text_val, text_test = random_split(text_dataset, [text_train_size, text_val_size, text_test_size])
    image_train, image_val, image_test = random_split(image_dataset, [image_train_size, image_val_size, image_test_size])

    text_train_loader = DataLoader(text_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    text_val_loader = DataLoader(text_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    text_test_loader = DataLoader(text_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    image_caption_train_loader = DataLoader(image_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    image_caption_val_loader = DataLoader(image_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    image_caption_test_loader = DataLoader(image_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (text_train_loader, text_val_loader, text_test_loader,
            image_caption_train_loader, image_caption_val_loader, image_caption_test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualStreamTransformer with BERT embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=10, help="Total epochs (text-only + image-caption)")
    parser.add_argument("--total-steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=5000, help="Eval steps")
    parser.add_argument("--text-only-epochs", type=int, default=5, help="Epochs to train on text-only data")
    parser.add_argument("--image-caption-epochs", type=int, default=5, help="Epochs to train on image-caption data")
    parser.add_argument("--d-model", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--checkpoint-dir", type=str, default="/local/scratch/bmg44/dual_stream_runs/checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint")
    args = parser.parse_args()

    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    vocab_size = len(tokenizer)

    print("Loading DINO embeddings and captions...")
    dino_embeddings, captions = load_and_concatenate_dino_data()

    print("Loading text-only data...")
    text_only_data = load_and_concatenate_text_only_data("/home/bmg44/DualStreamTransformer/data/text_only/processed_train_50M")

    (text_train_loader, text_val_loader, text_test_loader,
     image_caption_train_loader, image_caption_val_loader, image_caption_test_loader) = create_dataloaders(
         tokenizer=tokenizer,
         text_only_data=text_only_data,
         dino_embeddings=dino_embeddings,
         captions=captions,
         batch_size=args.batch_size,
         num_workers=4,
         seed=42
    )


    print(f"Text-only Train samples: {len(text_train_loader.dataset)}")
    # print(f"Image-caption Train samples: {len(image_caption_train_loader.dataset)}")

    print("Initializing DualStreamTransformer...")

    
    model = DualStreamTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_head=4,
        d_hid=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dino_dim=768,
        dropout=0.1
    )


    model = model.to(args.device)


    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        text_train_loader=text_train_loader,
        image_caption_train_loader=image_caption_train_loader,
        text_val_loader=text_val_loader,
        image_caption_val_loader=image_caption_val_loader,
        text_test_loader=text_test_loader,
        image_caption_test_loader=image_caption_test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.total_steps,
        text_only_epochs=args.text_only_epochs,
        image_caption_epochs=args.image_caption_epochs,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        clip_grad_norm=args.clip_grad_norm,
        eval_steps=args.eval_steps,
        wandb_project="dual-stream-model",
        wandb_run_name=f"dual_stream_{timestamp}",
        warmup_steps=args.warmup_steps
    )

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    print(f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    print(f"Training completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import os
import torch
import argparse
import datetime
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from torch.utils.data import Subset
from torch.utils.data import DataLoader, random_split
from combined_dataset import *
from model import DualStreamTransformer
from trainers.trainer_combined import Trainer
from transformers import AutoTokenizer
from utils import load_and_concatenate_dino_data, load_and_concatenate_text_only_data
from tokenizers.processors import TemplateProcessing
import json

import random
import numpy as np

def set_global_seed(seed, deterministic=False):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     if deterministic:
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False

def create_combined_dataloaders(
    tokenizer,
    text_data: list,
    dino_embeddings,
    captions: list,
    batch_size: int = 32,
    num_workers: int = 4,
    per_item_nums=(1, 1),
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    indices_file: str = "./data/combined_indices.json",
    save_indices: bool = True
):

    txt_ds = RawTextDataset(text_data)
    cap_ds = RawCaptionDataset(dino_embeddings, captions)

    combined = PerItemCombineDataset(
        datasets=[cap_ds, txt_ds],
        per_item_nums=list(per_item_nums),
        rd_ep_idx_order=True
    )

    total_len = len(combined)
    n_train = int(total_len * train_frac)
    n_val   = int(total_len * val_frac)
    n_test  = total_len - n_train - n_val

    if indices_file and os.path.exists(indices_file):
        with open(indices_file, 'r') as f:
            idxs = json.load(f)
        train_idx = idxs['train']
        val_idx   = idxs['val']
        test_idx  = idxs['test']
    else:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total_len, generator=gen).tolist()
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]

        if save_indices and indices_file:
            os.makedirs(os.path.dirname(indices_file), exist_ok=True)
            with open(indices_file, 'w') as f:
                json.dump({
                    'train': train_idx,
                    'val':   val_idx,
                    'test':  test_idx
                }, f)

    train_ds = Subset(combined, train_idx)
    val_ds   = Subset(combined, val_idx)
    test_ds  = Subset(combined, test_idx)

    collate = CombineCollate(
        tokenizer=tokenizer,
        name_prefixs=["cap_", "txt_"],
        add_image_pfx=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualStreamTransformer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=7000, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=20, help="Total epochs (text-only + image-caption)")
    parser.add_argument("--total-steps", type=int, default=722760, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=50000, help="Eval steps")
    parser.add_argument("--checkpoint-steps", type=int, default=50000, help="Checkpoint steps")
    parser.add_argument("--text-only-epochs", type=int, default=5, help="Epochs to train on text-only data")
    parser.add_argument("--image-caption-epochs", type=int, default=5, help="Epochs to train on image-caption data")
    parser.add_argument("--d-model", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-hid", type=int, default=3072, help="Number of hidden dimensions")
    parser.add_argument("--num-encoder-layers", type=int, default=5, help="Number of image encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=8, help="Number of decoder layers")
    parser.add_argument("--checkpoint-dir", type=str, default="/local/scratch/bmg44/dual_stream_runs/checkpoints/combined", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint")
    args = parser.parse_args()

    set_global_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = len(tokenizer)

    print("Loading DINO embeddings and captions...")
    dino_embeddings, captions = load_and_concatenate_dino_data()

    print("Loading text-only data...")
    text_only_data = load_and_concatenate_text_only_data("/home/bmg44/DualStreamTransformer/data/text_only/train_50M")

    train_loader, val_loader, test_loader = create_combined_dataloaders(
         tokenizer=tokenizer,
         text_data=text_only_data,
         dino_embeddings=dino_embeddings,
         captions=captions,
         batch_size=args.batch_size,
         num_workers=4,
         seed=42
    )


    print(f"Train samples: {len(train_loader.dataset)}")
    # print(f"Image-caption Train samples: {len(image_caption_train_loader.dataset)}")

    print("Initializing DualStreamTransformer...")

    
    model = DualStreamTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        d_hid=args.d_hid,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dino_dim=768,
        dropout=args.dropout
    )


    model = model.to(args.device)


    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.total_steps,
        text_only_epochs=args.text_only_epochs,
        image_caption_epochs=args.image_caption_epochs,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        clip_grad_norm=args.clip_grad_norm,
        eval_steps=args.eval_steps,
        checkpoint_steps=args.checkpoint_steps,
        wandb_project="dual-stream-model",
        wandb_run_name=f"dual_stream_{timestamp}",
        warmup_steps=args.warmup_steps,
        tokenizer=tokenizer
    )

    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    print(f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    print(f"Training completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import os
import torch
import argparse
import datetime
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from torch.utils.data import DataLoader, random_split
from datasets_def import TextOnlyDataset, DINOCaptionDataset
from models.model_film_on_text_image import DualStreamTransformer
from trainer import Trainer
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

def create_dataloaders(tokenizer, text_only_data, dino_embeddings, captions, batch_size=32, num_workers=4, seed=42, indices_file="./data/indices.json", save_indices=True):
    text_dataset = TextOnlyDataset(text_only_data, tokenizer)
    image_dataset = DINOCaptionDataset(dino_embeddings, captions, tokenizer)

    train_split, val_split = 0.8, 0.1
    text_train_size = int(len(text_dataset) * train_split)
    text_val_size = int(len(text_dataset) * val_split)
    text_test_size = len(text_dataset) - text_train_size - text_val_size

    image_train_size = int(len(image_dataset) * train_split)
    image_val_size = int(len(image_dataset) * val_split)
    image_test_size = len(image_dataset) - image_train_size - image_val_size

    if indices_file and torch.cuda.is_available():
        try:
            with open(indices_file, 'r') as f:
                indices = json.load(f)
            
            # Create splits using loaded indices
            text_train = torch.utils.data.Subset(text_dataset, indices['text_train'])
            text_val = torch.utils.data.Subset(text_dataset, indices['text_val'])
            text_test = torch.utils.data.Subset(text_dataset, indices['text_test'])
            
            image_train = torch.utils.data.Subset(image_dataset, indices['image_train'])
            image_val = torch.utils.data.Subset(image_dataset, indices['image_val'])
            image_test = torch.utils.data.Subset(image_dataset, indices['image_test'])
        except FileNotFoundError:
            print(f"Indices file {indices_file} not found. Creating new splits.")
            indices = None
    else:
        indices = None

    if indices is None:
        # Create new splits
        text_train, text_val, text_test = random_split(
            text_dataset, [text_train_size, text_val_size, text_test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        image_train, image_val, image_test = random_split(
            image_dataset, [image_train_size, image_val_size, image_test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        # Save indices if requested
        if save_indices and indices_file:
            indices = {
                'text_train': text_train.indices,
                'text_val': text_val.indices,
                'text_test': text_test.indices,
                'image_train': image_train.indices,
                'image_val': image_val.indices,
                'image_test': image_test.indices
            }
            with open(indices_file, 'w') as f:
                json.dump(indices, f)
            print(f"Saved split indices to {indices_file}")

    text_train_loader = DataLoader(text_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    text_val_loader = DataLoader(text_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    text_test_loader = DataLoader(text_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    image_caption_train_loader = DataLoader(image_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    image_caption_val_loader = DataLoader(image_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    image_caption_test_loader = DataLoader(image_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (text_train_loader, text_val_loader, text_test_loader,
            image_caption_train_loader, image_caption_val_loader, image_caption_test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualStreamTransformer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=12000, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=20, help="Total epochs (text-only + image-caption)")
    parser.add_argument("--total-steps", type=int, default=1107020, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=50000, help="Eval steps")
    parser.add_argument("--checkpoint-steps", type=int, default=50000, help="Checkpoint steps")
    parser.add_argument("--text-only-epochs", type=int, default=10, help="Epochs to train on text-only data")
    parser.add_argument("--image-caption-epochs", type=int, default=10, help="Epochs to train on image-caption data")
    parser.add_argument("--d-model", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-hid", type=int, default=3072, help="Number of hidden dimensions")
    parser.add_argument("--num-encoder-layers", type=int, default=5, help="Number of image encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=8, help="Number of decoder layers")
    parser.add_argument("--checkpoint-dir", type=str, default="/local/scratch/bmg44/dual_stream_runs/checkpoints/film_on_text_image", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint")
    args = parser.parse_args()

    set_global_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]', 'bos_token': '[BOS]'})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
    single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
    special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
    )
    vocab_size = len(tokenizer)

    print("Loading DINO embeddings and captions...")
    dino_embeddings, captions = load_and_concatenate_dino_data()

    print("Loading text-only data...")
    text_only_data = load_and_concatenate_text_only_data("/home/bmg44/DualStreamTransformer/data/text_only/train_50M")

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
        checkpoint_steps=args.checkpoint_steps,
        wandb_project="dual-stream-model",
        wandb_run_name=f"dual_stream_{timestamp}",
        warmup_steps=args.warmup_steps
    )

    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    print(f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    print(f"Training completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

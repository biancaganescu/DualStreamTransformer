import os
import torch
import argparse
import datetime
from torch.utils.data import DataLoader, random_split
from datasets_def import TextOnlyDataset, DINOCaptionDataset, UnifiedDataset
from model import DualStreamTransformer
from trainer import Trainer
from transformers import AutoTokenizer
from utils import load_and_concatenate_dino_data, load_and_concatenate_text_only_data
from tokenizers.processors import TemplateProcessing
import json

def create_unified_dataloaders(tokenizer, text_only_data, dino_embeddings, captions, batch_size=32, num_workers=4, seed=42, indices_file="./data/unified_indices.json", save_indices=True):
    # Create unified dataset
    dataset = UnifiedDataset(
        text_data=text_only_data,
        dino_embeddings=dino_embeddings,
        captions=captions,
        tokenizer=tokenizer,
        sequence_length=128
    )

    # Calculate split sizes
    train_split, val_split = 0.8, 0.1
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size

    if indices_file and torch.cuda.is_available():
        try:
            with open(indices_file, 'r') as f:
                indices = json.load(f)
            
            # Create splits using loaded indices
            train_dataset = torch.utils.data.Subset(dataset, indices['train'])
            val_dataset = torch.utils.data.Subset(dataset, indices['val'])
            test_dataset = torch.utils.data.Subset(dataset, indices['test'])
        except FileNotFoundError:
            print(f"Indices file {indices_file} not found. Creating new splits.")
            indices = None
    else:
        indices = None

    if indices is None:
        # Create new splits
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        # Save indices if requested
        if save_indices and indices_file:
            indices = {
                'train': train_dataset.indices,
                'val': val_dataset.indices,
                'test': test_dataset.indices
            }
            with open(indices_file, 'w') as f:
                json.dump(indices, f)
            print(f"Saved split indices to {indices_file}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualStreamTransformer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=12000, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=10, help="Total epochs (text-only + image-caption)")
    parser.add_argument("--total-steps", type=int, default=1107020, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=50000, help="Eval steps")
    parser.add_argument("--checkpoint-steps", type=int, default=50000, help="Checkpoint steps")
    parser.add_argument("--text-only-epochs", type=int, default=5, help="Epochs to train on text-only data")
    parser.add_argument("--image-caption-epochs", type=int, default=5, help="Epochs to train on image-caption data")
    parser.add_argument("--d-model", type=int, default=768, help="Model embedding dimension")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-hid", type=int, default=3072, help="Number of hidden dimensions")
    parser.add_argument("--num-encoder-layers", type=int, default=5, help="Number of image encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=8, help="Number of decoder layers")
    parser.add_argument("--checkpoint-dir", type=str, default="/local/scratch/bmg44/dual_stream_runs/checkpoints/medium_size", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from checkpoint")
    args = parser.parse_args()

    
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

    # (text_train_loader, text_val_loader, text_test_loader,
    #  image_caption_train_loader, image_caption_val_loader, image_caption_test_loader) = create_dataloaders(
    #      tokenizer=tokenizer,
    #      text_only_data=text_only_data,
    #      dino_embeddings=dino_embeddings,
    #      captions=captions,
    #      batch_size=args.batch_size,
    #      num_workers=4,
    #      seed=42
    # )
    train_loader, val_loader, test_loader = create_unified_dataloaders( tokenizer=tokenizer,
         text_only_data=text_only_data,
         dino_embeddings=dino_embeddings,
         captions=captions,
         batch_size=args.batch_size,
         num_workers=4,
         seed=42)


    # print(f"Text-only Train samples: {len(text_train_loader.dataset)}")
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
        unified=True,
        unified_train_loader=train_loader,
        unified_val_loader=val_loader,
        unified_test_loader=test_loader,
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

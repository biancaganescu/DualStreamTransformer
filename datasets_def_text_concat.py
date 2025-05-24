import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class TextOnlyDataset(Dataset):
    def __init__(self, text_data: str, tokenizer, sequence_length: int = 128):
        # 1. Tokenize the *entire* text as one long sequence (no special tokens)
        encoded_sentences = [
            tokenizer(
                sentence,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors="pt",
            )
            for sentence in text_data
        ]
        
        all_input_ids = torch.cat(
            [enc["input_ids"].squeeze(0) for enc in encoded_sentences],
            dim=0
        )  # shape: (total_tokens,)

        total_tokens = all_input_ids.size(0)
        n_blocks = total_tokens // sequence_length

        # cut any extra tokens and reshape into (n_blocks, sequence_length)
        input_ids = all_input_ids[: n_blocks * sequence_length]
        input_ids = input_ids.view(n_blocks, sequence_length)

        attention_masks = torch.ones_like(input_ids)

        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "text_input": self.input_ids[idx],
            "text_mask":  self.attention_masks[idx],
        }


class DINOCaptionDataset(Dataset):

    def __init__(self, dino_embeddings, captions, tokenizer, max_length=128):
        if len(dino_embeddings) != len(captions):
            raise ValueError(
                f"Mismatch: {len(dino_embeddings)} DINO embeddings vs {len(captions)} captions"
            )
        self.dino_embeddings = dino_embeddings 
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        dino_embedding = torch.tensor(self.dino_embeddings[idx], dtype=torch.float)
        
        encoding = self.tokenizer(caption, return_tensors="pt", truncation=True, max_length=self.max_length, padding='max_length', add_special_tokens=False)
        
        
        text_input = encoding.input_ids.squeeze(0)
        
        text_mask = encoding.attention_mask.squeeze(0)
   
        
        return {
            "text_input": text_input,
            "text_mask": text_mask,
            "dino_embedding": dino_embedding,
        }

class UnifiedDataset(Dataset):
    def __init__(self, text_data=None, dino_embeddings=None, captions=None, tokenizer=None, sequence_length=128, dino_dim=768):
        # Initialize lists if None
        self.text_data = text_data if text_data is not None else []
        self.dino_embeddings = dino_embeddings if dino_embeddings is not None else []
        self.captions = captions if captions is not None else []
        self.dino_dim = dino_dim
        
        # Validate DINO embeddings and captions match
        if len(self.dino_embeddings) != len(self.captions):
            raise ValueError(
                f"Mismatch: {len(self.dino_embeddings)} DINO embeddings vs {len(self.captions)} captions"
            )
        
        # Ensure tokenizer is provided
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Create a unified list of indices and types
        self.data = []
        # Add text-only data
        for i in range(len(self.text_data)):
            self.data.append(('text', i))
        # Add DINO-caption data
        for i in range(len(self.captions)):
            self.data.append(('dino', i))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_type, data_idx = self.data[idx]
        
        if data_type == 'text':
            # Handle text-only data
            text = self.text_data[data_idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.sequence_length,
                padding='max_length',
                return_tensors="pt",
                add_special_tokens=True
            )
            return {
                "text_input": encoding['input_ids'].squeeze(0),
                "text_mask": encoding['attention_mask'].squeeze(0),
                "dino_embedding": torch.zeros(self.dino_dim, dtype=torch.float)
            }
        else:
            # Handle DINO-caption data
            caption = self.captions[data_idx]
            dino_embedding = torch.tensor(self.dino_embeddings[data_idx], dtype=torch.float)
            
            encoding = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.sequence_length,
                padding='max_length',
                return_tensors="pt",
                add_special_tokens=True
            )
            
            return {
                "text_input": encoding['input_ids'].squeeze(0),
                "text_mask": encoding['attention_mask'].squeeze(0),
                "dino_embedding": dino_embedding
            }
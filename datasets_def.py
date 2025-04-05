import torch
from torch.utils.data import Dataset

class TextOnlyDataset(Dataset):

    def __init__(self, text_data, tokenizer, sequence_length=64):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.tokenizer.bos_token
    
        joined_text = ""
        for text in text_data:
            joined_text += self.tokenizer.bos_token + text + self.tokenizer.eos_token
        
        self.all_tokens = self.tokenizer(joined_text, return_tensors="pt").input_ids.squeeze(0)
        self.num_sequences = max(1, (len(self.all_tokens) - 1) // self.sequence_length)

    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length

        end_idx = min(start_idx + self.sequence_length, len(self.all_tokens))
        text_input = self.all_tokens[start_idx:end_idx]

        #all 1s since there is no padding
        text_mask = torch.ones_like(text_input)

        return {"text_input": text_input, "text_mask": text_mask,}

class DINOCaptionDataset(Dataset):

    def __init__(self, dino_embeddings, captions, tokenizer, max_length=64):
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
        
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input = encoding.input_ids.squeeze(0)
        text_mask = encoding.attention_mask.squeeze(0)
        
        return {
            "text_input": text_input,
            "text_mask": text_mask,
            "dino_embedding": dino_embedding,
        }

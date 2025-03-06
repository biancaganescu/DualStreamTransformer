import torch
from torch.utils.data import Dataset

class TextOnlyDataset(Dataset):

    def __init__(self, text_data, tokenizer, max_length=64):
        self.text_data = text_data 
        self.tokenizer = tokenizer
        self.max_length = max_length
        

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = self.text_data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input = encoding.input_ids.squeeze(0)  # [max_length]
        text_mask = encoding.attention_mask.squeeze(0)  # [max_length]
        
        return {
            "text_input": text_input,
            "text_mask": text_mask,
        }

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

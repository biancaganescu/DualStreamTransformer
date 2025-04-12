import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class TextOnlyDataset(Dataset):
    def __init__(self, text_data, tokenizer, sequence_length=128):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = self.text_data[idx]
        
        # Tokenize the text with special tokens
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.sequence_length,
            padding='max_length',
            return_tensors="pt",
            add_special_tokens=True  # Keep special tokens (CLS, SEP)
        )
        
        # Extract input_ids and attention_mask
        text_input = encoding['input_ids'].squeeze(0)
        text_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            "text_input": text_input,
            "text_mask": text_mask,
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
        
        encoding = self.tokenizer(caption, return_tensors="pt", truncation=True, max_length=self.max_length, padding='max_length', add_special_tokens=True)
        
        
        text_input = encoding.input_ids.squeeze(0)
        
        text_mask = encoding.attention_mask.squeeze(0)
   
        
        return {
            "text_input": text_input,
            "text_mask": text_mask,
            "dino_embedding": dino_embedding,
        }

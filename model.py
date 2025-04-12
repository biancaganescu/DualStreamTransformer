import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class DualStreamTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_head: int = 6,
        d_hid: int = 768,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dino_dim: int = 768,
        dropout: float = 0.1
    ):

        super().__init__()
        self.d_model = d_model

        # Embedding layers
        self.text_embedding = SimpleTextEmbedding(vocab_size, d_model)
        self.image_embedding = DinoImageEmbedding(dino_dim, d_model)

        # Image
        self.image_encoder = Encoder(d_model, n_head, d_hid, num_encoder_layers, dropout)

        # Decoder
        self.decoder = MultimodalDecoder(d_model, n_head, d_hid, num_decoder_layers, dropout)

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)


    def forward(
        self,
        input_ids,
        dino_embedding=None,
        padding_mask=None,
        use_image: bool = False
    ):

        # Text Embedding
        embedded = self.text_embedding(input_ids)

        # Image Embedding + Encoding (if use_image)
        if use_image:
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(image_embedded)
        else:
            image_encoded = None

        seq_len = embedded.size(1)

        # Causal mask for decoder
        tgt_mask = self.decoder.generate_square_subsequent_mask(seq_len).to(embedded.device)

        # Decoder pass
        decoder_output = self.decoder(tgt=embedded, image_memory=image_encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)

        output = self.output_layer(decoder_output)

        return output


    
    def generate(self, idx, dino_embedding=None, max_len=30, temperature=1.0, use_image=False, top_k=None, tokenizer=None):
 
        self.eval()
        device = idx.device
        generated = idx.clone()
        
        # Process image embeddings if provided
        image_encoded = None
        if use_image and dino_embedding is not None:
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(image_embedded)
        
        with torch.no_grad():
            for _ in range(max_len):
                # Get embeddings for current sequence
                logits = self.forward(
                input_ids=generated,
                dino_embedding=dino_embedding,
                use_image=use_image
                )
                
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Compute probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Debug output if tokenizer provided
                if tokenizer is not None:
                    topk_values, topk_indices = torch.topk(probs, 10, dim=-1)
                    top_tokens = tokenizer.convert_ids_to_tokens(topk_indices[0].tolist())
                    top_probs = topk_values[0].tolist()
                    print(f"Step {_ + 1}: Top 10 predictions:")
                    for token, prob in zip(top_tokens, top_probs):
                        print(f"  {token}: {prob:.4f}")
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if end token is generated for all sequences
                if tokenizer is not None and (next_token == tokenizer.sep_token_id).all():
                    break
        
        self.train()  # Restore training mode
        return generated




class SimpleTextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        

    def forward(self, x):
        batch_size, seq_len = x.size()
 
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        scale = math.sqrt(self.d_model)

        token_emb = self.token_embedding(x) * scale
        pos_emb = self.position_embedding(positions)
 
        embeddings = self.dropout(token_emb + pos_emb)
    
        return self.layer_norm(embeddings)


# Needs projection
class DinoImageEmbedding(nn.Module):
    def __init__(self, dino_dim, d_model):
        super().__init__()
        self.projection_layer = nn.Linear(dino_dim, d_model)

    def forward(self, x):
        return self.projection_layer(x.unsqueeze(1))


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return self.encoder(src, src_mask, src_key_padding_mask)


class DynamicGating(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate_fc = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, text_features, image_features):

        if image_features is None:
            return text_features

        combined = torch.cat([text_features, image_features], dim=-1)

        gate = torch.sigmoid(self.gate_fc(combined))

        fused = gate * text_features + (1 - gate) * image_features

        fused = self.layer_norm(self.dropout(fused))

        return fused


class MultimodalDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        # Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        # Cross Attention with Image
        self.cross_attn_image = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Gating + Fustion Module
        self.gate = DynamicGating(d_model, dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_model),
            nn.Dropout(dropout))

    def forward(self, tgt, image_memory, tgt_mask=None, tgt_key_padding_mask=None):
        # 1. Masked Self-Attention (causal)
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask, is_causal=True)
    
        tgt = self.norm1(tgt + self.dropout(self_attn_output))

        # 2. Cross-Attention to image + Gated Fusion
        if image_memory is not None:
            image_attn_output, _ = self.cross_attn_image(tgt, image_memory, image_memory)
            image_attn_output = self.dropout(image_attn_output)

            fused = self.gate(tgt, image_attn_output)
            tgt = self.norm2(tgt + fused)  


        # 3. Feedforward 
        ff_output = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt

    
class MultimodalDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([MultimodalDecoderLayer(d_model, n_head, d_hid, dropout) for _ in range(n_layers)])

    def generate_square_subsequent_mask(self, size):
        # mask = torch.triu(torch.ones(size, size), diagonal=1)
        # mask = mask.float().masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, float(0.0))
        # return mask
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


    def forward(self, tgt, image_memory, tgt_mask,  tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, image_memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return output

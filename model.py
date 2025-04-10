import torch
import torch.nn as nn
import torch.nn.functional as functional
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
        self.image_embedding = DinoImageEmbedding()

        # Dual-stream encoders
        self.text_encoder = Encoder(d_model, n_head, d_hid, num_encoder_layers, dropout)
        self.image_encoder = Encoder(d_model, n_head, d_hid, num_encoder_layers, dropout)

        # Decoder
        self.decoder = MultimodalDecoder(d_model, n_head, d_hid, num_decoder_layers, dropout)

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)


    def forward(
        self,
        text_input,
        dino_embedding,
        tgt,
        text_padding_mask=None,
        image_padding_mask=None,
        tgt_padding_mask=None,
        is_image_available=None,
        text_attention_mask=None,
        use_image: bool = False
    ):

        # Text Embedding + Encoding
        text_embedded = self.text_embedding(text_input)
        text_encoded = self.text_encoder(text_embedded, src_key_padding_mask=text_padding_mask)

        # Image Embedding + Encoding (if use_image)
        if use_image:
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(image_embedded, src_key_padding_mask=image_padding_mask)
        else:
            image_encoded = None

        # Target embedding for decode using text embeddings
        tgt_embedded = self.text_embedding(tgt)
        tgt_len = tgt_embedded.size(1)

        # Causal mask for decoder
        tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        # Decoder pass
        decoder_output = self.decoder(tgt_embedded, text_encoded, image_encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                                    text_memory_key_padding_mask=text_padding_mask, image_memory_key_padding_mask=image_padding_mask)

        output = self.output_layer(decoder_output)

        return output


    
    def generate(self, text_input, dino_embedding=None, max_len=50, temperature=1, use_image=False, top_k=None, tokenizer=None):
        self.eval()
        device = text_input.device        
        generated = torch.tensor([[tokenizer.cls_token_id]], device=device)
        
        with torch.no_grad():
            text_embedded = self.text_embedding(text_input)
            text_encoded = self.text_encoder(text_embedded)
        
        if use_image and dino_embedding is not None:
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(image_embedded)
        else:
            image_encoded = None
        
        
        for i in range(max_len):
            # Get embeddings for the current sequence
            tgt_embedded = self.text_embedding(generated) 
            # Get sequence length from the sequence dimension
            tgt_len = tgt_embedded.size(1)
            
            # Create causal mask for the current sequence length
            tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len).to(device)
            
            # Decoder pass
            decoder_output = self.decoder(
                tgt_embedded,
                text_encoded,
                image_encoded,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                text_memory_key_padding_mask=None,
                image_memory_key_padding_mask=None
            )
            
            
            last_output = decoder_output[:, -1, :] 
            
            logits = self.output_layer(last_output) 
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)

            # For debug
            topk_values, topk_indices = torch.topk(probs, 10, dim=-1) 
            top_tokens = tokenizer.convert_ids_to_tokens(topk_indices[0].tolist())
            top_probs = topk_values[0].tolist()
            print(f"Step {i+1}: Top 10 predictions:")
            for token, prob in zip(top_tokens, top_probs):
                print(f"  {token}: {prob:.4f}")

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the new token to the sequence
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == tokenizer.sep_token_id).all():
                break
        self.train()
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
        scale = torch.sqrt(torch.FloatTensor([self.d_model])).to(x.device)

        token_emb = self.token_embedding(x) * scale
        pos_emb = self.position_embedding(positions)
 
        embeddings = self.dropout(token_emb + pos_emb)
    
        return embeddings



class DinoImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(1)


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
        # Cross Attention with Text
        self.cross_attn_text = nn.MultiheadAttention( d_model, n_head, dropout=dropout, batch_first=True)
        # Cross Attention with Image
        self.cross_attn_image = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

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

    def forward(self, tgt, text_memory, image_memory,
                    tgt_mask=None, tgt_key_padding_mask=None,
                    text_memory_key_padding_mask=None, image_memory_key_padding_mask=None):

        tgt = self.norm1(self.dropout(self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,  key_padding_mask=tgt_key_padding_mask, is_causal=True)[0])) + tgt

        tgt = self.norm2(self.dropout(self.cross_attn_text(tgt, text_memory, text_memory, key_padding_mask=text_memory_key_padding_mask)[0]))

        tgt = self.norm3(self.dropout(self.ff(tgt)) + tgt)

        return tgt
        # tgt_norm = self.norm1(tgt)
        # tgt2, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm, 
        #                         key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask, is_causal=True)

        # # Residual connection
        # tgt = tgt + self.dropout(tgt2)

        # # Pre-layer normalization for text cross-attention
        # tgt_norm = self.norm2(tgt)
        # text_attn, _ = self.cross_attn_text(tgt_norm, text_memory, text_memory, 
        #                                    key_padding_mask=text_memory_key_padding_mask)
        # # Residual connection
        # text_out = tgt + self.dropout(text_attn)

        # # Pre-layer normalization for image cross-attention (if available)
        # if image_memory is not None:
        #     tgt_norm = self.norm3(tgt)
        #     image_attn, _ = self.cross_attn_image(tgt_norm, image_memory, image_memory, 
        #                                          key_padding_mask=image_memory_key_padding_mask)
        #     # Residual connection
        #     image_out = tgt + self.dropout(image_attn)

    
        # # Consistent computational path whether using image or not
        # if image_memory is None:
        #     fused = text_out
        # else:
        #     fused = self.gate(text_out, image_out)  # Use the same tensor for both since image features are already merged

        # # Pre-layer normalization for feed-forward
        # fused_norm = self.norm4(fused)
        # output = self.ff(fused_norm)
        
        # # Final residual connection
        # return fused + self.dropout(output)
        # # return self.ff(tgt2)

    
class MultimodalDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, n_head, d_hid, dropout, batch_first=True) for _ in range(n_layers)])

    def generate_square_subsequent_mask(self, size):
        # mask = torch.triu(torch.ones(size, size), diagonal=1)
        # mask = mask.float().masked_fill(mask == 1, float(-1e4)).masked_fill(mask == 0, float(0.0))
        # return mask
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


    def forward(self, tgt, text_memory, image_memory, tgt_mask,  tgt_key_padding_mask=None, text_memory_key_padding_mask=None, image_memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            # output = layer(output, text_memory, image_memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            #                text_memory_key_padding_mask=text_memory_key_padding_mask, image_memory_key_padding_mask=image_memory_key_padding_mask)

            output = layer(output, text_memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=text_memory_key_padding_mask, tgt_is_causal=True)


        return output

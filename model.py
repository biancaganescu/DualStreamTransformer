import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from transformers import BertModel


class DualStreamTransformer(nn.Module):
    """
    Args:
    d_model: the number of expected features in the encoder/decoder inputs (default=768).
    nhead: the number of heads in the multiheadattention models (default=6).
    num_encoder_layers: the number of sub-encoder-layers in the encoder (default=4).
    num_decoder_layers: the number of sub-decoder-layers in the decoder (default=4).        
    dropout: the dropout value (default=0.1).
    """

    def __init__(
        self,
        vocab_size: int,
        bert_model_name: str = "bert-base-uncased",
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
        # self.text_embedding = BertTextEmbedding(bert_model_name)
        self.text_embedding = SimpleTextEmbedding(vocab_size, d_model)
        self.image_embedding = DinoImageEmbedding()

        # Dual-stream encoders
        self.text_encoder = Encoder(
            d_model, n_head, d_hid, num_encoder_layers, dropout)
        self.image_encoder = Encoder(
            d_model, n_head, d_hid, num_encoder_layers, dropout)

        # Decoder
        self.decoder = MultimodalDecoder(
            d_model, n_head, d_hid, num_encoder_layers, dropout)

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        """
            Forward pass:
            text_input:         token IDs for BERT
            dino_embedding:     image embeddings (DINO)
            tgt:                decoder input tokens
            is_image_available: optional flag for whether images are present
        """

        # Text Embedding + Encoding
        text_embedded = self.text_embedding(
            text_input)
        text_encoded = self.text_encoder(
            text_embedded, src_key_padding_mask=text_padding_mask)

        # Image Embedding + Encoding (if use_image)
        if use_image:
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(
                image_embedded, src_key_padding_mask=image_padding_mask)
        else:
            image_encoded = None

        # Target embedding for decode using text embeddings
        tgt_embedded = self.text_embedding(tgt)
        tgt_len = tgt_embedded.size(0)

        # Causal mask for decoder
        tgt_mask = self.decoder.generate_square_subsequent_mask(
            tgt_len).to(tgt.device)

        # Decoder pass
        decoder_output = self.decoder(tgt_embedded, text_encoded, image_encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                                    text_memory_key_padding_mask=text_padding_mask, image_memory_key_padding_mask=image_padding_mask)

        output = self.output_layer(decoder_output.transpose(0, 1))

        return output

    def generate(self, text_input, dino_embedding=None, max_len=50, temperature=1.0, BOS_TOKEN_ID=101, EOS_TOKEN_ID=102, text_attention_mask=None, use_image=False):
        self.eval()

        device = text_input.device
        batch_size = text_input.size(0)

        # Encode Text Input
        text_embedded = self.text_embedding(
            text_input)
        text_encoded = self.text_encoder(
            text_embedded, src_key_padding_mask=None)

        # Encode Image Input
        if use_image and dino_embedding is not None:
            image_embedded = self.image_embedding(dino_embedding)
            image_encoded = self.image_encoder(
                image_embedded, src_key_padding_mask=None)
        else:
            image_encoded = None

        for _ in range(max_len - 1):

            tgt_embedded = self.text_embedding(generated)
            tgt_len = tgt_embedded.size(0)  # [tgt_seq_len, batch, d_model]

            # Causal mask
            tgt_mask = self.decoder.generate_square_subsequent_mask(
                tgt_len).to(device)

            # Decoder pass
            decoder_output = self.decoder(tgt_embedded, text_encoded, image_encoded, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=None, text_memory_key_padding_mask=None, image_memory_key_padding_mask=None)

            last_output = decoder_output[-1]  # [batch, d_model]
            logits = self.output_layer(last_output)  # [batch, vocab_size]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            generated = torch.cat([generated, next_token], dim=1)

            # If all sequences have generated EOS, stop early
            if (next_token == EOS_TOKEN_ID).all():
                break

        self.train()
        return generated

class SimpleTextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.d_model = d_model

    def forward(self, x):
        batch_size, seq_len = x.size()

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        embeddings = (token_emb + pos_emb) * math.sqrt(self.d_model)

        embeddings = self.dropout(embeddings)
        
        return embeddings.transpose(0, 1) 

class BertTextEmbedding(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

    def forward(self, x, attention_mask=None):
        x = self.bert.embeddings(input_ids=x)
        x *= math.sqrt(self.bert_dim)
        return x.transpose(0, 1)  # [seq_len, batch, d_model]


class DinoImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(0)  # [1, batch_size, dino_dim]


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_hid, dropout, activation="gelu")
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

        # potentially change to self.layer_norm(text_features + self.dropout(fused))
        fused = self.layer_norm(self.dropout(fused))

        return fused


class MultimodalDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        # Self Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout)
        # Cross Attention with Text
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout)
        # Cross Attention with Image
        self.cross_attn_image = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout)

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
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, tgt, text_memory, image_memory,
                    tgt_mask=None, tgt_key_padding_mask=None,
                    text_memory_key_padding_mask=None, image_memory_key_padding_mask=None):
        """
        Args:
            tgt: [tgt_seq_len, batch_size, d_model]
            text_memory: [text_seq_len, batch_size, d_model]
            image_memory: [1, batch_size, d_model] or [img_seq_len, batch_size, d_model]
        """

        # Masked self-attention
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        tgt = self.norm1(self.dropout(tgt2) + tgt)

        # Cross-attention to text
        text_attn, _ = self.cross_attn_text(
            tgt, text_memory, text_memory, key_padding_mask=text_memory_key_padding_mask)

        text_out = self.norm2(self.dropout(text_attn) + tgt)

        # Cross-attention to image
        if image_memory is not None:
            image_attn, _ = self.cross_attn_image(
                tgt, image_memory, image_memory, key_padding_mask=image_memory_key_padding_mask)

            image_out = self.norm3(self.dropout(image_attn) + tgt)
        else:
            image_out = None

        # Gating
        fused = self.gate(text_out, image_out)

        # Feed forward for the output
        output = self.ff(fused)

        return self.norm4(output + fused)


class MultimodalDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([MultimodalDecoderLayer(
            d_model, n_head, d_hid, dropout) for _ in range(n_layers)])

    def generate_square_subsequent_mask(self, size):
        """
        Generates an upper-triangular mask for causal decoding.
        """
        mask = torch.triu(torch.ones(size, size)) == 1
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, text_memory, image_memory, tgt_mask,  tgt_key_padding_mask=None, text_memory_key_padding_mask=None, image_memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, text_memory, image_memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                           text_memory_key_padding_mask=text_memory_key_padding_mask, image_memory_key_padding_mask=image_memory_key_padding_mask)

        return output

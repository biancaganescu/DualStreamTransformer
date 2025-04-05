from model import DualStreamTransformer
from transformers import BertTokenizerFast
import torch

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Pad token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)
vocab_size = len(tokenizer)

text = "Hello, my name is "
# Get the tokenized input as a dictionary
tokenized_input = tokenizer(text, return_tensors="pt")
# Extract the input_ids and move to GPU
input_ids = tokenized_input['input_ids'].to("cuda:0")

model = DualStreamTransformer(
    vocab_size=vocab_size,
    d_model=768,
    n_head=12,
    d_hid=768,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dino_dim=768,
    dropout=0.1
)
model.to("cuda:0")  # Make sure the model is also on the GPU

model.load_state_dict(torch.load("/local/scratch/bmg44/dual_stream_runs/checkpoints/run_20250316_224841/best_checkpoint.pt")["model_state_dict"])
print("global step ", torch.load("/local/scratch/bmg44/dual_stream_runs/checkpoints/run_20250316_224841/best_checkpoint.pt")["global_step"])
# Now pass the tensor to generate
generated = model.generate(input_ids)
print(tokenizer.decode(generated[0]))
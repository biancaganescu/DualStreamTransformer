from model import DualStreamTransformer
from transformers import BertTokenizerFast
import torch

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

vocab_size = len(tokenizer)

text = "[CLS] Where did you go? "
# Get the tokenized input as a dictionary
tokenized_input = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# Extract the input_ids and move to GPU
input_ids = tokenized_input['input_ids'].to("cuda:0")
model = DualStreamTransformer(
    vocab_size=vocab_size,
    d_model=768,
    n_head=4,
    d_hid=512,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dino_dim=768,
    dropout=0.1
    # vocab_size=vocab_size,
    # d_model=768,
    # n_head=12,
    # d_hid=768,
    # num_encoder_layers=6,
    # num_decoder_layers=6,
    # dino_dim=768,
    # dropout=0.1
)
model.to("cuda:0")  # Make sure the model is also on the GPU

model.load_state_dict(torch.load("/local/scratch/bmg44/dual_stream_runs/checkpoints/run_20250410_125059/best_checkpoint.pt")["model_state_dict"])
print("global step ", torch.load("/local/scratch/bmg44/dual_stream_runs/checkpoints/run_20250410_125059/best_checkpoint.pt")["global_step"])
# Now pass the tensor to generate
print(input_ids)
generated = model.generate(input_ids, temperature=1, tokenizer=tokenizer)
print(tokenizer.decode(generated[0]))
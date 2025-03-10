import json
import numpy as np
import os
def load_and_concatenate_dino_data():
    caption_files = ["./data/image_caption/cc_3M_captions.json", "./data/image_caption/local_narr_captions.json"]

    all_captions = []
    for caption_file in caption_files:
        with open(caption_file, "r") as f:
            captions = json.load(f)
        all_captions.extend(captions)
    
    processed_embeddings = np.load("./data/image_caption/processed_embeddings.npy")

    processed_embeddings = processed_embeddings / (np.linalg.norm(processed_embeddings, axis=1, keepdims=True) + 1e-8)

    assert len(all_captions) == processed_embeddings.shape[0], (
        f"Mismatch: {len(all_captions)} captions vs {processed_embeddings.shape[0]} embeddings"
    )
    
    print(f"DINO embeddings loaded with shape: {processed_embeddings[0].shape}")
    
    return processed_embeddings, all_captions

def load_and_concatenate_text_only_data(directory):
  
    all_texts = []
    
    for filename in sorted(os.listdir(directory)):  
        if filename.endswith(".train"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines() 
                all_texts.extend(lines)
    
    print("all texts snippet ", all_texts[:5])
    return all_texts



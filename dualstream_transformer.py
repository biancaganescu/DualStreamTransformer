import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Dict, Any
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, clear_torch_cache
from lm_eval import utils
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import os
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

from model import DualStreamTransformer

@register_model("dualstream")
class DualStreamLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 128

    def __init__(self,
    model_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: Optional[Union[int, str]] = 1,
    max_batch_size: Optional[int] = None,
    max_length: int = 128,
    image_src: Optional[str] = None,
    image_src_split: Optional[str] = None,
    image_key: str = "image",
    type: str = None,
    cache_embeddings: bool = False,
    ):

        super().__init__()

        # Initialise tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]', 'bos_token': '[BOS]'})
        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=self.tokenizer.bos_token + " $A " + self.tokenizer.eos_token,
        special_tokens=[(self.tokenizer.eos_token, self.tokenizer.eos_token_id), (self.tokenizer.bos_token, self.tokenizer.bos_token_id)],
        )
        vocab_size = len(self.tokenizer)
        self.vocab_size = vocab_size
        

        # Initialise model
        if model_path is None:
            raise ValueError("Model path must be provided")

        checkpoint = torch.load(model_path)

        if type == "large":
            self._model = DualStreamTransformer(
            vocab_size=self.vocab_size,
            d_model=1024,
            n_head=16,
            d_hid=4096,
            num_encoder_layers=6,
            num_decoder_layers=8,
            dino_dim=768,
            dropout=0.1
            )
        elif type == "medium":
            self._model = DualStreamTransformer(
            vocab_size=self.vocab_size,
            d_model=768,
            n_head=12,
            d_hid=3072,
            num_encoder_layers=4,
            num_decoder_layers=6,
            dino_dim=768,
            dropout=0.1
            )
        else:
            model_args = checkpoint.get("model_args", {})
            if not model_args:
                    raise ValueError("Checkpoint does not contain model_args")
            self._model = DualStreamTransformer(**model_args)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(device=device)
        self._model.eval()
    

        # Inialise image to DINO embeddings processor
        self.dino_model = AutoModel.from_pretrained("facebook/dino-vitb16").to(device)
        self.dino_processor = AutoProcessor.from_pretrained("facebook/dino-vitb16")
        self.dino_model.eval()

        self._max_length = max_length
        self.truncation_max_length = max_length
        self.batch_size_per_gpu = int(batch_size) if isinstance(batch_size, str) else batch_size
        self._device = torch.device(device)
        self.image_src = image_src
        self.image_src_split = image_src_split
        self.image_key = image_key
        self.cache_embeddings = cache_embeddings
        self._rank = 0
        self._world_size = 1


    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.tokenizer.pad_token_id
            return self.tokenizer.pad_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 128

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return self.tokenizer.pad_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device
    
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=True
    ) -> List[int]:

        special_tokens_kwargs = {}
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": True}  # Matches training
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(
            string,
            truncation=True,
            max_length=self.truncation_max_length,
            **special_tokens_kwargs
        )

        if left_truncate_len is not None:
                encoding = encoding[-left_truncate_len:]

        return encoding


    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        texts = []
        images = []

        for s in strings:
            if isinstance(s, dict) and self.image_key in s:
                texts.append(s.get("text", ""))
                images.append(s[self.image_key])
            else:
                texts.append(s)
                images.append(None)
            
        encoding = self.tokenizer(
            texts,
            padding="longest",
            truncation=truncation or True,
            max_length=self.truncation_max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        if left_truncate_len is not None:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        dino_embeddings = None
        if any(img is not None for img in images):
            valid_images = [img for img in images if img is not None]
            inputs = self.dino_processor(images=[img.convert('RGB') for img in valid_images], return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                dino_embeddings = outputs.last_hidden_state[:, 0, :]  # [num_images, 768]
            full_embeddings = torch.zeros(len(strings), 768, device=self._device)
            img_idx = 0
            for i, img in enumerate(images):
                if img is not None:
                    full_embeddings[i] = dino_embeddings[img_idx]
                    img_idx += 1
            dino_embeddings = full_embeddings
        
        return input_ids, attention_mask, dino_embeddings

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


    def _model_call(
        self,
        inputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dino_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        padding_mask = attn_mask.eq(0) if attn_mask is not None else None
        use_image = dino_embedding is not None

        with torch.no_grad():
            logits = self._model(
                input_ids=inputs,
                dino_embedding=dino_embedding,
                padding_mask=padding_mask,
                use_image=use_image,
            )

        return logits


    def _model_generate(self, context, max_length, stop, dino_embedding: Optional[torch.Tensor] = None, **generation_kwargs):
        temperature = generation_kwargs.get("temperature", 1.0)
        top_k = generation_kwargs.get("top_k", None)
        do_sample = generation_kwargs.get("do_sample", True)

        # Handle temperature for greedy decoding
        if do_sample is False or temperature == 0.0:
            temperature = 1.0
            top_k = 1  # Greedy decoding

        use_image = dino_embedding is not None
        generated = self._model.generate(
            idx=context,
            dino_embedding=dino_embedding,
            max_len=max_length - context.size(1) if context.size(1) > 0 else max_length,
            temperature=temperature,
            use_image=use_image,
            top_k=top_k,
            tokenizer=self.tokenizer
        )
        
        return generated

    def _select_cont_toks(self, logits: torch.Tensor, contlen: int, inplen: int) -> torch.Tensor:
        """Select continuation tokens from logits for scoring."""
        if contlen:
            # For causal models, we score only the continuation tokens
            # This should return exactly contlen logits
            logits = logits[inplen - contlen:inplen]
        return logits


    def loglikelihood_rolling(self, requests: List[Instance], disable_tqdm: bool = False) -> List[float]:
        """
        Compute rolling log-likelihoods over full sequences.
        """
        loglikelihoods = []

        adaptive_batch_size = None

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood_rolling requests"
        ):
            # Check for image data
            dino_embedding = None
            if isinstance(string, dict) and self.image_key in string:
                if self.dino_model is not None:
                    dino_embedding = self._load_dino_embedding(string[self.image_key])
                string = string.get("text", "")

          # Get rolling windows to compute perplexity
            token_list = self.tok_encode(string)
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            
            # Add None for context and DINO embedding
            rolling_token_windows = [(None, [], tokens, dino_embedding) for _, tokens in rolling_token_windows]
            
            # Compute log-likelihoods for each window
            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
                disable_tqdm=True)
            
            # Sum log probabilities (discarding is_greedy values)
            string_nll = sum(x[0] for x in string_nll)
            
            loglikelihoods.append(string_nll)
        
        return loglikelihoods

    # def _loglikelihood_tokens(
    #     self,
    #     requests: List[Tuple[Optional[str], List[int], List[int], Optional[Any]]],
    #     disable_tqdm: bool = False,
    #     override_bs: int = None,
    # ) -> List[Tuple[float, bool]]:


    #     res = []

    #     def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
    #         """Defines the key for the sorted method"""
    #         # the negative sign on len(toks) sorts descending - this has a few advantages:
    #         # - time estimates will always be over not underestimates, which is more useful for planning
    #         # - to know the size of a batch when going through the list, you know the first one is always the batch
    #         #   padded context length. this is useful to simplify the batching logic and more importantly to make
    #         #   automatic adaptive batches much much easier to implement
    #         # - any OOMs will happen right away rather than near the end

    #         toks = req[1] + req[2]
    #         return -len(toks), tuple(toks)

    #     def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
    #         """Defines the key to group and lookup one-token continuations"""
    #         # Use with group_by="contexts" (optional)"
    #         # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
    #         # speeds up some multiple-choice tasks proportionally to the number of choices.
    #         # groups requests by context+continuation[:-1] and infer on one request/group.
    #         return list(req[-len(req)+1]) + req[-len(req)+2][:-1]

    #     re_ord = Collator(
    #         requests,
    #         sort_fn=_collate,
    #         # group_by="contexts"
    #         group_fn=_lookup_one_token_cont,
    #     )

    #     n_reordered_requests = len(re_ord)

    #     batch_size = self.batch_size_per_gpu

    #     chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

    #     pbar = tqdm(
    #         total=len(requests),
    #         disable=(disable_tqdm or (self.rank != 0)),
    #         desc="Running loglikelihood requests",
    #     )

    #     for chunk in chunks:
    #         inps = []
    #         cont_toks_list = []
    #         inplens = []
    #         dino_embeddings = []

    #         padding_len_inp = 0
    #         for _, context_enc, continuation_enc, image in chunk:
    #             assert len(context_enc) > 0
    #             assert len(continuation_enc) > 0
    #             assert len(continuation_enc) <= self.max_length

    #             if any(tok < 0 or tok >= self.vocab_size for tok in continuation_enc):
    #                 raise ValueError(f"Invalid tokens in continuation_enc: {continuation_enc}")
    #             inp = torch.tensor(
    #                 (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
    #                 dtype=torch.long,
    #                 device=self._device,
    #             )
    #             (inplen,) = inp.shape
    #             inps.append(inp)
    #             cont_toks_list.append(continuation_enc)
    #             inplens.append(inplen)
    #             padding_len_inp = max(padding_len_inp, inplen)

    #             # Handle image
    #             dino_embedding = None
    #             if image is not None:
    #                 dino_embedding = self._load_dino_embedding(image)
    #             dino_embeddings.append(dino_embedding)

    #         # Pad inputs
    #         batched_inps = pad_and_concat(
    #             padding_len_inp, inps, padding_side="right"
    #         )
    #         batched_attn_mask = pad_and_concat(
    #             padding_len_inp, [torch.ones_like(inp) for inp in inps], padding_side="right"
    #         )

    #         # Stack DINO embeddings
    #         dino_emb = None
    #         if any(de is not None for de in dino_embeddings):
    #             dino_emb = torch.stack([
    #                 de if de is not None else torch.zeros(self._model.dino_dim, device=self._device)
    #                 for de in dino_embeddings
    #             ])

    #         # Forward pass
    #         multi_logits = F.log_softmax(
    #             self._model_call(batched_inps, attn_mask=batched_attn_mask, dino_embedding=dino_emb),
    #             dim=-1
    #         )

    #         for i, (cont_toks, inplen) in enumerate(zip(cont_toks_list, inplens)):
    #             contlen = len(cont_toks)
    #             if contlen == 0:
    #                 res.append((0.0, True))
    #                 pbar.update(1)
    #                 continue
    #             logits = multi_logits[i]
    #             # Compute ctx_len for consistency with HFLM
    #             ctx_len = inplen 

    #             logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)

    #             logits = logits.unsqueeze(0)
    #             cont_toks_tensor = torch.tensor(cont_toks, dtype=torch.long, device=self._device)

    #             # Check for exact match with greedy tokens
    #             greedy_tokens = logits.argmax(dim=-1)  # [1, seq_len]
    #             cont_toks_tensor_expanded = cont_toks_tensor.unsqueeze(0)  # [1, seq_len]
    #             is_greedy = (greedy_tokens == cont_toks_tensor_expanded).all().item()
                
    #             # Get log probabilities using gather on dimension 2 (vocabulary dimension)
    #             # We need to make cont_toks shape [1, seq_len, 1] for gather
    #             gather_indices = cont_toks_tensor.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
                
    #             # Make sure token IDs don't exceed vocabulary size
    #             vocab_size = logits.size(2)
    #             if (gather_indices >= vocab_size).any():
    #                 print(f"Warning: Token IDs exceed vocab size! Max token: {gather_indices.max().item()}, Vocab size: {vocab_size}")
    #                 # Clip token IDs to valid range
    #                 gather_indices = torch.clamp(gather_indices, 0, vocab_size - 1)
                
    #             selected_logits = torch.gather(logits, 2, gather_indices).squeeze(-1)  # [1, seq_len]
    
    #             loglikelihood = selected_logits.sum().item()


    #             res.append((loglikelihood, is_greedy))
    #             pbar.update(1)

    #     pbar.close()
    #     return re_ord.get_original(res)
        
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Optional[str], List[int], List[int], Optional[Any]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        """Core implementation of log-likelihood calculation."""
        res = []

        def _collate(req):
            """Sort by descending sequence length for efficient batching."""
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(
            requests,
            sort_fn=_collate,
        )

        batch_size = override_bs if override_bs is not None else self.batch_size_per_gpu
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []
            dino_embeddings = []

            for cache_key, context_enc, continuation_enc, image in chunk:
                # For empty inputs, provide default
                if len(context_enc) == 0:
                    context_enc = [self.prefix_token_id]
                
                assert len(continuation_enc) > 0, "Continuation cannot be empty"
                assert len(continuation_enc) <= self.max_length, "Continuation too long"

                # Concatenate context and continuation, then trim to max_length
                full_seq = (context_enc + continuation_enc)[-(self.max_length + 1):]
                # Remove last token (will be predicted)
                inp = torch.tensor(full_seq[:-1], dtype=torch.long, device=self._device)
                
                inplen = len(inp)
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
                
                # Handle the image embedding if present
                if image is not None:
                    dino_embedding = self._load_dino_embedding(image)
                    dino_embeddings.append(dino_embedding)
                else:
                    dino_embeddings.append(None)

            # Pad sequences for batch processing
            padding_len = max(len(inp) for inp in inps)
            batched_inps = pad_and_concat(padding_len, inps, padding_side="right")
            
            # Create attention masks (1 for tokens, 0 for padding)
            batched_attn_mask = torch.ones_like(batched_inps, device=self._device)
            for i, inp_len in enumerate(inplens):
                batched_attn_mask[i, inp_len:] = 0
            
            # Process images if available
            dino_emb = None
            if any(de is not None for de in dino_embeddings):
                # Create a batch of image embeddings
                dino_emb = torch.stack([
                    de if de is not None else torch.zeros(768, device=self._device)
                    for de in dino_embeddings
                ])

            # Get model logits
            multi_logits = F.log_softmax(
                self._model_call(batched_inps, attn_mask=batched_attn_mask, dino_embedding=dino_emb),
                dim=-1
            )

            # Process each example in the batch
            for i, (cont_toks, inplen) in enumerate(zip(cont_toks_list, inplens)):
                contlen = len(cont_toks)
                
                # Get logits for this example
                logits = multi_logits[i]
                
                # Select the relevant continuation tokens
                # We need to determine which logits correspond to continuation tokens
                if len(cont_toks) > inplen:
                    # If continuation is longer than input, we can only score the beginning
                    contlen = inplen
                    cont_toks = cont_toks[:contlen]
                    
                # Calculate starting position for continuation logits
                start_pos = max(0, inplen - contlen)
                
                # Extract logits for continuation tokens only
                cont_logits = logits[start_pos:inplen]
                
                # Convert to tensor for easier manipulation
                cont_toks_tensor = torch.tensor(cont_toks[-len(cont_logits):], dtype=torch.long, device=self._device)
                
                # Check if generated tokens match continuation (greedy match)
                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_toks_tensor).all().item()
                
                # Get log probabilities for the actual continuation tokens
                indices = cont_toks_tensor.unsqueeze(1)  # [seq_len, 1]
                
                # Ensure indices are valid (within vocab size)
                vocab_size = cont_logits.size(1)
                if (indices >= vocab_size).any():
                    indices = torch.clamp(indices, 0, vocab_size - 1)
                
                # Gather the log probabilities of the continuation tokens
                token_logprobs = torch.gather(cont_logits, 1, indices).squeeze(1)  # [seq_len]
                loglikelihood = token_logprobs.sum().item()

                res.append((loglikelihood, is_greedy))
                
                # Add to cache if key provided
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, (loglikelihood, is_greedy))
                    
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

        
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []
        batch_size = self.batch_size_per_gpu

        def _collate(req: Tuple[Union[str, Dict], Dict]):
            toks = self.tok_encode(req[0] if isinstance(req[0], str) else req[0]["text"])
            return -len(toks), req[0]

        re_ords = Collator(
            [req.args for req in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self._rank != 0)),
            desc="Running generate_until requests",
        )


        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]  # Assume same kwargs in batch
            kwargs = gen_kwargs.copy()

            until = kwargs.pop("until", [self.tok_decode(self.eot_token_id, skip_special_tokens=False)])
            if isinstance(until, str):
                until = [until]
            until.append(self.tok_decode(self.eot_token_id, skip_special_tokens=False))

            max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
            max_ctx_len = self.max_length - max_gen_toks

            # Encode contexts
            input_ids, attn_masks, dino_emb = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=True,
            )


            if "max_length" not in kwargs:
                this_max_len = input_ids.shape[1] + max_gen_toks
            else:
                this_max_len = kwargs["max_length"]

            # Generate
            cont = self._model_generate(
                context=input_ids,
                max_length=max_gen_toks,
                stop=until,
                dino_embedding=dino_emb,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                cont_toks = cont_toks[len(self.tok_encode(context if isinstance(context, str) else context["text"])):]
                s = self.tok_decode(cont_toks, skip_special_tokens=True)

                for term in until:
                    if term:
                        s = s.split(term)[0]

                res.append(s)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
        
    def _load_dino_embedding(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Load or compute DINO embedding for an image."""
        if isinstance(image, str):
            # Check for cached embedding
            if self.image_src and image.endswith('.npy'):
                embedding_path = os.path.join(self.image_src, image)
                if os.path.exists(embedding_path):
                    return torch.tensor(np.load(embedding_path), device=self._device)
            image_path = image
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                if self.image_src and not os.path.isabs(image_path):
                    if self.image_src_split:
                        image_path = os.path.join(self.image_src, self.image_src_split, image_path)
                    else:
                        image_path = os.path.join(self.image_src, image_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found at {image_path}")
                img = Image.open(image_path).convert('RGB')
        else:
            img = image.convert('RGB')

        # Check cache
        cache_path = None
        if self.cache_embeddings and isinstance(image, str) and self.image_src:
            cache_name = os.path.splitext(os.path.basename(image))[0] + '.npy'
            cache_path = os.path.join(self.image_src, cache_name)
            if os.path.exists(cache_path):
                return torch.tensor(np.load(cache_path), device=self._device)

        # Compute DINO embedding
        inputs = self.dino_processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        # Save to cache if enabled
        if self.cache_embeddings and cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, cls_token.cpu().numpy())

        return cls_token.squeeze(0)  # [768]

    def loglikelihood(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        batch_size = self.batch_size_per_gpu

        def _collate(req: Instance):
            context, continuation = req.args
            ctx_tokens = self.tok_encode(context if isinstance(context, str) else context["text"])
            cont_tokens = self.tok_encode(continuation, add_special_tokens=False)
            return -len(ctx_tokens + cont_tokens), tuple(ctx_tokens + cont_tokens)

        re_ord = Collator([req for req in requests], sort_fn=_collate)
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self._rank != 0)),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            contexts = []
            continuations = []
            images = []

            for req in chunk:
                assert req.request_type == "loglikelihood"
                context, continuation = req.args
                contexts.append(context)
                continuations.append(continuation)
                image = context.get(self.image_key) if isinstance(context, dict) else None
                images.append(image)
        
        
            requests_chunk = [
                (None,
                 self.tok_encode(ctx if isinstance(ctx, str) else ctx["text"]),
                 self.tok_encode(cont, add_special_tokens=False),
                 ctx.get(self.image_key) if isinstance(ctx, dict) else None)
                for ctx, cont in zip(contexts, continuations)
            ]
            chunk_res = self._loglikelihood_tokens(requests_chunk, disable_tqdm=True)
            res.extend(chunk_res)
            pbar.update(len(chunk))

        pbar.close()
        return re_ord.get_original(res)
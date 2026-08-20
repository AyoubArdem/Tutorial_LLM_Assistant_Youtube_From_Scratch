# Welcome to the complete, step-by-step tutorial on building a custom Large Language Model (LLM) from scratch!

In this comprehensive guide, we're going to dive deep into the Colab notebook you are currently exploring. Our journey will cover the construction of an **Encoder-Decoder Transformer** architecture, pre-training on massive corpora like Wikipedia, and fine-tuning on instruction-following datasets.

Let's dissect the code and understand the *rationale* behind every single design and implementation choice.

---

## Phase 1: The Setup & Vocabulary (Tokenizer)

Before any language model can process text, it must first convert human-readable words into a numerical format that computers can understand. This process is handled by a **tokenizer**. Instead of reinventing the wheel, we leverage the time-tested GPT-2 tokenizer.

```python
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Helps debug PyTorch CUDA errors by forcing synchronous execution
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 1. Load the GPT-2 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
```

### Why this step matters: Choosing and Configuring a Tokenizer

We opt for the `GPT-2 tokenizer` due to its widespread adoption and proven effectiveness. It employs a **Byte-Pair Encoding (BPE)** algorithm, which efficiently breaks down words into small, meaningful subunits. This approach balances vocabulary size and model complexity.

A crucial configuration is setting `tokenizer.pad_token = tokenizer.eos_token`. GPT-2's tokenizer doesn't have a default padding token (a special token used to make all sequences in a batch the same length). By assigning it to the end-of-sequence token, we ensure all padding tokens are treated uniformly during training.

---

## Phase 2: Building the Brain (The Transformer Architecture)

This section defines our custom Large Language Model, named `LLM`, as a PyTorch `nn.Module`. Unlike solely decoder-only architectures like standard GPT models, our notebook implements an **Encoder-Decoder Transformer**, similar to the original "Attention is All You Need" paper.

- **The Encoder**: Processes and understands the input context, such as a YouTube transcript or a user's instruction. It generates a rich, context-aware representation of the input.
- **The Decoder**: Takes the encoder's representation and generates the response word by word, conditioned on the input context.

```python
class LLM(nn.Module):
    def __init__(self, vocab_size=tokenizer.vocab_size, d_model=768, nhead=8, num_layers=5, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embeddings: Converting Token IDs into High-Dimensional Vectors
        #   - tok_embedding: Maps each token ID to a dense vector (d_model dimensions).
        #   - pos_embedding: Learns positional information, crucial since Transformers process tokens in parallel.
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 2. The Encoder Stack: 5 layers of self-attention for input understanding
        #   - d_model: The dimensionality of the input and output of the sub-layers in the model.
        #   - nhead: The number of attention heads in the multi-head attention models.
        #   - batch_first=True: Ensures input tensors are (batch_size, sequence_length, features).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. The Decoder Stack: 5 layers to generate text based on the Encoder's output
        #   - Similar parameters to the encoder, but designed for generation.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. The Output Layer: Projects the Decoder's output back to the vocabulary size
        #   - This linear layer predicts the probability distribution over all possible tokens.
        self.fc_out = nn.Linear(d_model, vocab_size)

    def embed(self, input_ids):
        # Combines the token's semantic meaning with its sequential position.
        # The (self.d_model ** 0.5) scaling factor is a common practice to prevent
        # the positional embeddings from dominating the token embeddings, improving stability.
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        return (self.tok_embedding(input_ids) * (self.d_model ** 0.5)) + self.pos_embedding(positions)
```

### The Magic of Causal Masking in the `forward` pass

A critical component for text generation within the Decoder is the **causal mask**. When the Decoder generates the next word in a sequence, it must *not* have access to subsequent (future) tokens in the target sequence. This ensures genuine autoregressive generation.

```python
    def forward(self, input_ids_encoder, attention_mask_encoder, input_ids_decoder, attention_mask_decoder):
        # 1. Embed inputs using the custom embedding function.
        encoder_embedded = self.embed(input_ids_encoder)
        decoder_embedded = self.embed(input_ids_decoder)
        
        # 2. Generates a diagonal mask for the decoder. 
        #    Tokens can only attend to themselves and previous tokens. 
        #    -inf values effectively "block" attention to future tokens.
        tgt_seq_len = input_ids_decoder.size(1)
        tgt_mask = (
            nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
            .to(dtype=torch.float32, device=input_ids_decoder.device)
            .masked_fill(
                nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
                .to(device=input_ids_decoder.device)
                .bool(),
                float("-inf"),
            )
        )

        # 3. Create padding masks for both encoder and decoder.
        #    These masks tell the Transformer to ignore padding tokens during attention calculations.
        src_key_padding_mask = (attention_mask_encoder == 0).bool()
        tgt_key_padding_mask = (attention_mask_decoder == 0).bool()

        # 4. Encoder pass: Processes the input and generates a context-rich representation.
        encoder_output = self.transformer_encoder(
            encoder_embedded, src_key_padding_mask=src_key_padding_mask
        )

        # 5. Decoder pass: Generates the output sequence, attending to the encoder's output
        #    (memory) and its own causally masked sequence.
        decoder_output = self.transformer_decoder(
            tgt=decoder_embedded,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 6. Final linear layer projects decoder output to vocabulary logits.
        return self.fc_out(decoder_output)
```

---

## Phase 3: Unsupervised Pre-training (Learning to Speak)

The first major training phase involves **unsupervised pre-training**. The goal here is to teach the model the fundamental structures, grammar, and vast factual knowledge embedded within the English language and world at large.

A key challenge with such large datasets is memory management. Downloading the entire Wikipedia dataset into RAM would easily crash your Colab session. To circumvent this, the notebook brings in the power of **streaming datasets**.

```python
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# Stream the dataset over the internet chunk-by-chunk, efficiently handling large data.
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
```

This `streaming=True` argument allows the dataset to be loaded and processed in smaller chunks on-the-fly, preventing memory overflows.

### The Pre-training Objective: Learning Context and Continuity

During the pre-training loop, a specific objective is employed: for each input sequence (a piece of a Wikipedia article), you truncate it at approximately the 85% mark. The **Encoder** receives the first 85%, and the **Decoder** is tasked with predicting the remaining 15%. This trains the model to:

- How to generate coherent text based on a given context.
- The underlying grammar and semantic relationships within the language.
- Factual knowledge present in the Wikipedia articles.

### Hardware Optimization Tricks (Mixed Precision & Gradient Accumulation)

To enable the training of this 144 million parameter model efficiently on commonly available free GPUs (like the NVIDIA T4 in Colab), two advanced optimization techniques are used in the training loop:

1. **`torch.amp.autocast` (Automatic Mixed Precision - AMP)**: This feature automatically performs operations in 16-bit floating-point (FP16) where appropriate, while maintaining 32-bit (FP32) precision where numerical stability is critical.

   Benefits:
   - **Halved Memory Usage**: FP16 tensors consume half the memory of FP32 tensors, allowing larger models or batch sizes.
   - **Increased Speed**: Modern GPUs have specialized hardware (Tensor Cores) that can perform FP16 calculations much faster.
   - **Stable Training**: The `GradScaler` component handles numerical stability issues that can arise from using lower precision.

2. **Gradient Accumulation**: Due to GPU memory constraints, you might only be able to fit a small number of examples (e.g., 4) in a single batch. Gradient accumulation addresses this by accumulating gradients over several small batches before performing a weight update.

   Benefits:
   - **Larger Effective Batch Size**: Improves the stability of gradient estimates and can lead to better generalization.
   - **Overcomes Memory Limits**: Allows training with a large *effective* batch size even when physical memory is limited.

---

## Phase 4: Instruction Fine-Tuning (Learning to Serve)

After pre-training, our model has a strong grasp of the English language, but it primarily knows how to complete sentences. To make it a helpful assistant, we need to teach it to follow instructions and provide helpful responses.

We use the **Databricks Dolly 15K dataset**, a curated collection of 15,000 instruction-response pairs across various domains. We format this data into a custom `InstructionDataset` class, ensuring it's structured correctly for our Encoder-Decoder model.

### The -100 Padding Trick: Optimizing Loss Calculation

A clever trick is used when preparing the `labels` for the loss function:

```python
labels = raw_dec_input_ids.clone()
labels[labels == self.tokenizer.pad_token_id] = -100 
```

*Why `-100`?* In PyTorch's `nn.CrossEntropyLoss` function, any target index with the value `-100` is **ignored** during loss calculation by default. This is incredibly useful for padding tokens—we want the model to learn from real tokens but not "learn" from padding tokens.

During fine-tuning, we also employ a **lower Learning Rate** (e.g., `2e-5`) compared to pre-training. This prevents **catastrophic forgetting**, where the model might overwrite its broad knowledge with narrow task-specific patterns.

---

## Phase 5: The YouTube Assistant Interface

The culmination of our model's development is its application as a YouTube Assistant, demonstrated by the `LLM_Assistant_Youtube` function. This function integrates our fine-tuned LLM with YouTube transcript retrieval capabilities, enabling real-time question answering about video content.

```python
# Example usage:
URL = "https://youtu.be/msFxQ7OYPj8?si=..." # Replace with actual URL
query = "how to learn AI from scratch"
result = LLM_Assistant_Youtube(URL, query)
print(f"Generated Answer: {result}")
```

Behind the scenes, the `LLM_Assistant_Youtube` function performs several key steps:

1. **Transcript Retrieval**: It takes a YouTube URL and uses `langchain_community.document_loaders.YoutubeLoader` to fetch the video's transcript and metadata. If no transcript is available, it gracefully handles the error.

2. **Context Chunking (if RAG is enabled, though not explicitly in this snippet)**: In a full RAG (Retrieval Augmented Generation) setup, the transcript would be split into overlapping chunks, embedded into vector space, and the most relevant chunks would be retrieved based on the user's query.

3. **Prompt Formatting**: It constructs a structured prompt using the user's `query` and the retrieved `context_text` (from the YouTube transcript), following the pattern: `"Instruction: {query}\nContext: {context_text}\nAnswer:"`.

4. **Input Tokenization**: The formatted prompt is tokenized using our `tokenizer` and converted into input IDs and attention masks, truncated to the model's `max_length` of 512 tokens.

5. **Response Generation**: The tokenized input is fed to the `generate_response_text` function. This function uses the `fine_tuned_model` in evaluation mode (`model.eval()`) to generate the response autoregressively, one token at a time.

6. **Output Decoding**: The generated token IDs are then decoded back into a human-readable string, with special tokens removed for a clean output.

### Final Thoughts on Improvement and Scaling

As you've observed, while this custom LLM is a powerful demonstration, achieving performance comparable to state-of-the-art models like ChatGPT requires further scaling and refinement. Key areas for enhancement include:

1. **More Data:** Our pre-training on Wikipedia is foundational, but cutting-edge LLMs are trained on **terabytes** of diverse data, including web scrapes (like CommonCrawl), books, code repositories, and multilingual corpora.

2. **Larger Context Windows:** Our model uses a `max_seq_len` of 512 tokens, which is relatively small for processing long documents or conversations. Modern LLMs feature context windows ranging from 4K to 200K+ tokens.

3. **Increased Scale:** Our model with 144 million parameters is roughly the size of the original GPT-1. Contemporary LLMs range from billions (e.g., Llama 2 7B) to hundreds of billions (e.g., GPT-3 175B) of parameters.

4. **Advanced Decoding Strategies:** While greedy decoding with repetition penalty is simple, more sophisticated techniques like Top-K, Top-P (nucleus) sampling, or beam search can produce more diverse and higher-quality outputs.

5. **Reinforcement Learning from Human Feedback (RLHF):** A crucial step for aligning LLMs with human preferences and making them truly helpful is often achieved through RLHF, where the model learns to optimize for human-rated quality signals.

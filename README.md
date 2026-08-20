```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Building a Custom LLM from Scratch - Detailed Tutorial</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.7; color: #333; max-width: 900px; margin: 20px auto; padding: 0 20px; background-color: #f9f9f9; }
        h1, h2, h3 { color: #0056b3; margin-top: 1.5em; }
        h1 { border-bottom: 2px solid #0056b3; padding-bottom: 0.5em; }
        h3 { border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
        pre { background-color: #eef; border: 1px solid #ddd; border-left: 5px solid #0056b3; color: #444; page-break-inside: avoid; font-family: 'Consolas', 'Monaco', monospace; font-size: 14px; line-height: 1.5; margin-bottom: 1.6em; max-width: 100%; overflow: auto; padding: 1em 1.5em; display: block; word-wrap: break-word; border-radius: 5px; }
        code { font-family: 'Consolas', 'Monaco', monospace; background-color: #f0f0f0; padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        hr { border: 0; height: 1px; background: #ddd; margin: 2em 0; }
        strong { color: #222; }
        em { font-style: italic; color: #555; }
        ul, ol { margin-left: 20px; }
    </style>
</head>
<body>
    <h1>Welcome to the complete, step-by-step tutorial on building a custom Large Language Model (LLM) from scratch!</h1>

    <p>In this comprehensive guide, we're going to dive deep into the  Colab notebook you are currently exploring. Our journey will cover the construction of an **Encoder-Decoder Transformer** architecture, boasting approximately 144 million parameters. We will meticulously pre-train this model on the vast Wikipedia dataset to instill a foundational understanding of the English language. Subsequently, we will fine-tune it on the Databricks Dolly 15k dataset to empower it to follow human instructions, culminating in its application as an interactive YouTube Assistant.</p>

    <p>Let's dissect the code and understand the *rationale* behind every single design and implementation choice.</p>

    <hr>

    <h2>Phase 1: The Setup &amp; Vocabulary (Tokenizer)</h2>

    <p>Before any language model can process text, it must first convert human-readable words into a numerical format that computers can understand. This process is handled by a **tokenizer**. Instead of creating a tokenizer from scratch, which is a complex task, we leverage a highly optimized, pre-existing one.</p>

    <pre><code>import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Helps debug PyTorch CUDA errors by forcing synchronous execution
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 1. Load the GPT-2 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

</code></pre>

    <p><strong>Why this step matters: Choosing and Configuring a Tokenizer</strong></p>
    <p>We opt for the `GPT-2 tokenizer` due to its widespread adoption and proven effectiveness. It employs a **Byte-Pair Encoding (BPE)** algorithm, which efficiently breaks down words into smaller sub-word units (e.g., "playing" might become "play" and "ing"). This strategy is excellent for handling rare words and reducing vocabulary size while still representing a broad range of text.</p>
    <p>A crucial configuration is setting `tokenizer.pad_token = tokenizer.eos_token`. GPT-2's tokenizer doesn't have a default padding token (a special token used to make all sequences in a batch the same length). By assigning the "End of Sentence" (`eos_token`) to be the padding token, we ensure that all input sequences can be uniformly processed, which is essential for batching in neural networks.</p>

    <hr>

    <h2>Phase 2: Building the Brain (The Transformer Architecture)</h2>

    <p>This section defines our custom Large Language Model, named `LLM`, as a PyTorch `nn.Module`. Unlike solely decoder-only architectures like standard GPT models, our notebook implements an **Encoder-Decoder** Transformer. This architecture is reminiscent of models like T5 or BART and is particularly well-suited for sequence-to-sequence tasks where both an input context and a generated output sequence are important.</p>
    <ul>
        <li><strong>The Encoder</strong>: Processes and understands the input context, such as a YouTube transcript or a user's instruction. It generates a rich, context-aware representation of the input.</li>
        <li><strong>The Decoder</strong>: Takes the encoder's representation and generates the response word by word, conditioned on the input context.</li>
    </ul>

    <pre><code>class LLM(nn.Module):
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

</code></pre>

    <h3>The Magic of Causal Masking in the `forward` pass</h3>
    <p>A critical component for text generation within the Decoder is the **causal mask**. When the Decoder generates the next word in a sequence, it must *not* have access to subsequent (future) words. This constraint prevents information leakage and forces the model to predict tokens based solely on preceding ones. We enforce this in the `forward` pass:</p>

    <pre><code>    def forward(self, input_ids_encoder, attention_mask_encoder, input_ids_decoder, attention_mask_decoder):
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

</code></pre>

    <hr>

    <h2>Phase 3: Unsupervised Pre-training (Learning to Speak)</h2>

    <p>The first major training phase involves **unsupervised pre-training**. The goal here is to teach the model the fundamental structures, grammar, and vast factual knowledge embedded within the English language. We accomplish this by exposing it to a massive dataset: Wikipedia.</p>

    <p>A key challenge with such large datasets is memory management. Downloading the entire Wikipedia dataset into RAM would easily crash your Colab session. To circumvent this, the notebook brilliantly utilizes **Streaming**:</p>

    <pre><code>from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# Stream the dataset over the internet chunk-by-chunk, efficiently handling large data.
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

</code></pre>

    <p>This `streaming=True` argument allows the dataset to be loaded and processed in smaller chunks on-the-fly, preventing memory overflows.</p>

    <h3>The Pre-training Objective: Learning Context and Continuity</h3>
    <p>During the pre-training loop, a specific objective is employed: for each input sequence (a piece of a Wikipedia article), you truncate it at approximately the 85% mark. The **Encoder** receives the first 85% of the article, while the **Decoder** is tasked with predicting the *remaining 15%*. This forces the model to learn:</p>
    <ul>
        <li>How to generate coherent text based on a given context.</li>
        <li>The underlying grammar and semantic relationships within the language.</li>
        <li>Factual knowledge present in the Wikipedia articles.</li>
    </ul>

    <h3>Hardware Optimization Tricks (Mixed Precision &amp; Gradient Accumulation)</h3>
    <p>To enable the training of this 144 million parameter model efficiently on commonly available free GPUs (like the NVIDIA T4 in Colab), two advanced optimization techniques are used in the training loop:</p>
    <ol>
        <li><strong><code>torch.amp.autocast</code> (Automatic Mixed Precision - AMP)</strong>: This feature automatically performs operations in 16-bit floating-point (FP16) where appropriate, while retaining 32-bit floating-point (FP32) precision for sensitive operations (like loss calculation). The benefits are substantial:
            <ul>
                <li><strong>Halved Memory Usage</strong>: FP16 tensors consume half the memory of FP32 tensors, allowing larger models or batch sizes.</li>
                <li><strong>Increased Speed</strong>: Modern GPUs have specialized hardware (Tensor Cores) that can perform FP16 calculations much faster.</li>
                <li><strong>Stable Training</strong>: The `GradScaler` component handles numerical stability issues that can arise from using lower precision.</li>
            </ul>
        </li>
        <li><strong>Gradient Accumulation</strong>: Due to GPU memory constraints, you might only be able to fit a small number of examples (e.g., 4) in a single batch. Gradient accumulation addresses this by accumulating gradients over several mini-batches before performing a single optimizer step. In this notebook, accumulating gradients over 4 steps effectively simulates a larger batch size of 16, leading to:
            <ul>
                <li><strong>Larger Effective Batch Size</strong>: Improves the stability of gradient estimates and can lead to better generalization.</li>
                <li><strong>Overcomes Memory Limits</strong>: Allows training with a large *effective* batch size even when physical memory is limited.</li>
            </ul>
        </li>
    </ol>

    <hr>

    <h2>Phase 4: Instruction Fine-Tuning (Learning to Serve)</h2>

    <p>After pre-training, our model has a strong grasp of the English language, but it primarily knows how to complete sentences. To make it a helpful assistant, we need to teach it to follow instructions. This is where **instruction fine-tuning** comes in. We introduce the **Databricks Dolly 15k** dataset, which is specifically designed for this purpose, containing triplets of `Instruction`, `Context`, and `Response`.</p>

    <p>We format this data into a custom `InstructionDataset` class, ensuring it's structured correctly for our Encoder-Decoder model.</p>

    <h3>The -100 Padding Trick: Optimizing Loss Calculation</h3>
    <p>A clever trick is used when preparing the `labels` for the loss function:</p>

    <pre><code>labels = raw_dec_input_ids.clone()
labels[labels == self.tokenizer.pad_token_id] = -100 

</code></pre>

    <p><em>Why `-100`?</em> In PyTorch's `nn.CrossEntropyLoss` function, any target index with the value `-100` is **ignored** during loss calculation by default. This is incredibly useful for padding tokens. By setting padding tokens in the labels to `-100`, we prevent the model from wasting computational effort trying to predict these irrelevant tokens, focusing its learning on generating meaningful text.</p>

    <p>During fine-tuning, we also employ a **lower Learning Rate** (e.g., `2e-5`) compared to pre-training. This prevents **catastrophic forgetting**, where the model might overwrite its broad knowledge gained during pre-training. Instead, the lower learning rate encourages the model to *adapt* its existing knowledge to the new instruction-following task. Additionally, a **Cosine Scheduler with Warmup** is used, which smoothly increases the learning rate initially (warmup) to prevent early training instability, and then gradually decreases it following a cosine curve. This schedule is known to improve convergence and generalization performance.</p>

    <hr>

    <h2>Phase 5: The YouTube Assistant Interface</h2>

    <p>The culmination of our model's development is its application as a YouTube Assistant, demonstrated by the `LLM_Assistant_Youtube` function. This function integrates our fine-tuned LLM with tools to fetch and process YouTube video content.</p>

    <pre><code># Example usage:
URL = "https://youtu.be/msFxQ7OYPj8?si=..." # Replace with actual URL
query = "how to learn AI from scratch"
result = LLM_Assistant_Youtube(URL, query)
print(f"Generated Answer: {result}")

</code></pre>

    <p>Behind the scenes, the `LLM_Assistant_Youtube` function performs several key steps:</p>
    <ol>
        <li><strong>Transcript Retrieval</strong>: It takes a YouTube URL and uses `langchain_community.document_loaders.YoutubeLoader` to fetch the video's transcript and metadata. If no transcript is available or an error occurs (e.g., due to missing YouTube API key), it provides a placeholder.</li>
        <li><strong>Context Chunking (if RAG is enabled, though not explicitly in this snippet)</strong>: In a full RAG (Retrieval Augmented Generation) setup, the transcript would be split into smaller, manageable chunks using a `RecursiveCharacterTextSplitter`. These chunks would then be stored in a vector database (`Chroma`) and relevant chunks retrieved based on the query. Even without explicit RAG in the final function, the context is truncated to `1200` characters to fit within the model's input limits.</li>
        <li><strong>Prompt Formatting</strong>: It constructs a structured prompt using the user's `query` and the retrieved `context_text` (from the YouTube transcript), following the pattern: `Instruction: {query}\nContext: {context_text}\nAnswer:`</li>
        <li><strong>Input Tokenization</strong>: The formatted prompt is tokenized using our `tokenizer` and converted into input IDs and attention masks, truncated to the model's `max_length` of 512, and moved to the active `device` (GPU).</li>
        <li><strong>Response Generation</strong>: The tokenized input is fed to the `generate_response_text` function. This function uses the `fine_tuned_model` in evaluation mode (`model.eval()`) and iteratively predicts one token at a time until a maximum length (`max_length=100`) is reached or an End-of-Sentence token is generated. It employs a `repetition_penalty` to discourage repetitive outputs and uses greedy decoding (selecting the most probable token at each step).</li>
        <li><strong>Output Decoding</strong>: The generated token IDs are then decoded back into a human-readable string, with special tokens removed for a clean output.</li>
    </ol>

    <h3>Final Thoughts on Improvement and Scaling</h3>

    <p>As you've observed, while this custom LLM is a powerful demonstration, achieving performance comparable to state-of-the-art models like ChatGPT requires further scaling and refinement. Key areas for improvement include:</p>
    <ol>
        <li><strong>More Data:</strong> Our pre-training on Wikipedia is foundational, but cutting-edge LLMs are trained on **terabytes** of diverse data, including web scrapes (like CommonCrawl), books, and conversational data, over many more epochs.</li>
        <li><strong>Larger Context Windows:</strong> Our model uses a `max_seq_len` of 512 tokens, which is relatively small for processing long documents or conversations. Modern LLMs feature context windows of 8k, 32k, 128k, or even 1 million tokens, allowing them to grasp much broader contexts.</li>
        <li><strong>Increased Scale:</strong> Our model with 144 million parameters is roughly the size of the original GPT-1. Contemporary LLMs range from billions (e.g., Llama 2 7B) to hundreds of billions (e.g., GPT-3) and even trillions of parameters, which significantly enhances their capacity for understanding, reasoning, and generation.</li>
        <li><strong>Advanced Decoding Strategies:</strong> While greedy decoding with repetition penalty is simple, more sophisticated techniques like Top-K, Top-P (nucleus) sampling, or beam search can lead to more diverse, coherent, and higher-quality generated text.</li>
        <li><strong>Reinforcement Learning from Human Feedback (RLHF):</strong> A crucial step for aligning LLMs with human preferences and making them truly helpful is often achieved through RLHF, which involves human evaluation and reinforcement learning.</li>
    </ol>
</body>
</html>
```

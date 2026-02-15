# Inference Optimization

## 1. Throughput vs Latency
* **Latency**: Time to first token (TTFT). Optimized by lower precision and faster interconnects.
* **Throughput**: Tokens per second. Optimized by batching.

## 2. Flash Attention 2
The single most important optimization.
* Reduces memory complexity from $O(N^2)$ to linear.
* **Usage**: `model = AutoModel.from_pretrained(..., attn_implementation="flash_attention_2")`.

## 3. Quantization
* **AWQ/GPTQ**: Weight-only quantization. Good for compute-bound scenarios (batch size 1).
* **KV Cache Quantization (FP8)**: Crucial for long-context inference. Reduces cache memory by 2x, allowing 2x larger batch sizes.

## 4. Serving Engines (Do not use raw PyTorch loop)
* **vLLM**: State-of-the-art PagedAttention. Best for throughput.
* **TGI (Text Generation Inference)**: Hugging Face standard. Great Tensor Parallelism support.
* **SGLang**: Best for structured output (JSON enforcement).

---
license: apache-2.0
base_model: BAAI/bge-reranker-v2-m3
library_name: onnx
pipeline_tag: text-classification
tags:
  - onnx
  - fp16
  - reranker
  - djembed
---

# bge-reranker-v2-m3, ONNX fp16 packed

[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) at revision `953dc6f`, as an fp16 ONNX
Runtime graph in packing mode: fused attention that computes no padding. Built for
[Djembed](https://github.com/RAGuLabs/Djembed).

| | |
|---|---|
| Inputs | `input_ids`, `attention_mask` (int64), a (query, document) pair per row |
| Outputs | `logits` (1 per pair); relevance is its sigmoid |
| Runtime | ONNX Runtime CUDA execution provider only: `PackedAttention` has no CPU kernel |
| Tested with | `onnxruntime_gpu` 1.29 (CUDA 12.8, cuDNN 9) |
| Agreement | scores within 0.0032 of Text Embeddings Inference fp16 on the same inputs |

## Use with Djembed

```
hf download ragulabs-org/bge-reranker-v2-m3-onnx-fp16-packed --local-dir models/bge-reranker-v2-m3
```

```yaml
models:
  - name: bge-reranker-v2-m3
    task: rerank
    path: /models/bge-reranker-v2-m3
    device: cuda:0
```

## How it was built

`djembed-bench/export-packed.sh <dir> fp16` in the Djembed repository: exported with PyTorch 2.1.2 and transformers
4.44.2 (eager attention), fused and converted to fp16 by ONNX Runtime's transformer optimizer, then converted to
packing mode.

Apache 2.0, as the original model.

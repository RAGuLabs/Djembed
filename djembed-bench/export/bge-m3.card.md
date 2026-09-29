---
license: mit
base_model: BAAI/bge-m3
library_name: onnx
pipeline_tag: feature-extraction
tags:
  - onnx
  - fp16
  - sentence-embeddings
  - djembed
---

# bge-m3, ONNX fp16 packed

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) at revision `5617a9f`, dense embeddings, as an fp16 ONNX Runtime
graph in packing mode: fused attention that computes no padding. Built for
[Djembed](https://github.com/RAGuLabs/Djembed).

| | |
|---|---|
| Inputs | `input_ids`, `attention_mask` (int64) |
| Outputs | `sentence_embedding` (CLS pooled, L2-normalised, 1024), `token_embeddings` |
| Runtime | ONNX Runtime CUDA execution provider only: `PackedAttention` has no CPU kernel |
| Tested with | `onnxruntime_gpu` 1.29 (CUDA 12.8, cuDNN 9) |
| Agreement | cosine ≥ 0.99994 with Text Embeddings Inference fp16 on the same inputs |

## Use with Djembed

```
hf download ragulabs-org/bge-m3-onnx-fp16-packed --local-dir models/bge-m3
```

```yaml
models:
  - name: bge-m3
    task: embed
    path: /models/bge-m3
    device: cuda:0
```

## How it was built

`djembed-bench/export-packed.sh <dir> fp16` in the Djembed repository: exported with PyTorch 2.1.2 and transformers
4.44.2 (eager attention), fused and converted to fp16 by ONNX Runtime's transformer optimizer, then converted to
packing mode.

MIT, as the original model.

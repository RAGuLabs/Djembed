# AMD GPU support

Status: researched, not implemented. Everything below is verified against source code or official documentation;
what only real hardware can answer is listed at the end.

## Findings

### ONNX Runtime on AMD, from Java

- ONNX Runtime removed its ROCm execution provider in 1.23; AMD's supported path is MIGraphX
  ([ROCm EP docs](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)). Djembed uses 1.29.
- The published Maven jars carry native libraries for CPU, CUDA and TensorRT only.
- The Java API has no `addMIGraphX`, so ONNX Runtime's built-in MIGraphX provider cannot be enabled from Java.
- It does expose plugin execution providers: `OrtEnvironment.registerExecutionProviderLibrary`, `getEpDevices` and
  `SessionOptions.addExecutionProvider`.
- AMD's MIGraphX plugin is [onnxruntime/onnxruntime-ep-amdgpu](https://github.com/onnxruntime/onnxruntime-ep-amdgpu):
  it needs ONNX Runtime ≥ 1.24, has no releases, and is built from source (ROCm SDK, CMake ≥ 4.2, Python ≥ 3.12).

The plugin is the only route from Java without patching ONNX Runtime.

### Models

Checked against the MIGraphX ONNX parser (`ROCm/AMDMIGraphX`, `src/onnx/`, commit `67638f7`) and our own graphs.

- Supported contrib operators: `Attention`, `MultiHeadAttention`, `SkipLayerNormalization`, `LayerNormalization`,
  `BiasGelu`, `FastGelu`, `Gelu`, `BiasAdd`.
- Not supported: `PackedAttention`, `RemovePadding`, `RestorePadding`. Packed models (`*-fp16-packed`) cannot run.
- `Attention` accepts a raw 2D mask only; a 1D mask index throws "Left/Right Padding not currently supported"
  (`parse_attention.cpp:291–293`). Our fused models use a 1D mask index (`attention_mask` → `Cast` → `ReduceSum`).
- ONNX Runtime's optimizer emits the raw 2D mask with `--use_raw_attention_mask` (`fusion_options.py`).

An AMD model is the export recipe of `djembed-bench/export-packed.sh` with `--use_raw_attention_mask` and without
the packing step.

### Input shapes

Checked against the plugin source (`src/migraphx/mgx_ep.cc`, commit `7d4bda2`).

- Every new input shape triggers a full parse and compile of a new MIGraphX program (`ResolveProgram`), unless a
  compiled `.mxr` for that shape is in the cache directory (`migraphx_model_cache_dir`).
- At most 4 compiled programs stay in memory, least recently used evicted (`max_resident_programs`, `mgx_ep.h:347`).
- Bucketing options exist: batch rounded up to powers of two (`max_dynamic_batch`) or to listed sizes
  (`compile_batches`); sequence length padded to one fixed length (`static_pad_seq`, `static_pad_seq_len`,
  `static_pad_inputs`, `static_pad_outputs`).

Djembed produces a new `rows × length` shape on nearly every batch, so as it works today it would recompile
continuously. It must emit a small set of shapes instead. One fixed sequence length would bring back the padding cost
that packing removes on CUDA; a few length buckets bound it.

### Other servers

Infinity publishes AMD images (`michaelf34/infinity:0.0.77-amd`,
[write-up](https://huggingface.co/blog/michaelfeil/infinity-amd)). TEI's AMD support was not checked.

## Plan

1. **Plugin providers in core.** Config for the plugin library path and device selection, loaded through the API
   above. Generic, not AMD-specific.
2. **Shape buckets in the batch planner.** A few batch sizes × a few length buckets, precompiled at start-up and
   persisted as `.mxr` files; `max_resident_programs` sized to the bucket count.
3. **AMD export variant.** Fused fp16 with `--use_raw_attention_mask`, not packed.
4. **Docker image.** A ROCm base that builds the MIGraphX plugin.
5. **Benchmark** against Infinity's AMD image, with the same method as on CUDA.

Steps 1–3 can be written and unit-tested without AMD hardware; steps 4–5 and any performance claim need it.

## Open questions (hardware needed)

- Compile time of one program, which decides how many shape buckets are affordable.
- Throughput and latency against Infinity.
- Whether the plugin builds cleanly against the target ROCm version.

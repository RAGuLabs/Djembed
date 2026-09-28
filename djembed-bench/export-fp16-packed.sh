#!/usr/bin/env bash
# Builds the fp16 models of the benchmark's fp16 round: bge-m3 and bge-reranker-v2-m3, from the Hugging Face revisions
# TEI serves, exported with classic attention, fused and converted to fp16 by ONNX Runtime's transformer optimizer,
# then converted to its packing mode (PackedAttention: the encoder computes no padding).
#
#   djembed-bench/export-fp16-packed.sh <models dir>
#
# Creates <models dir>/bge-m3-fp16-packed and <models dir>/bge-reranker-v2-m3-fp16-packed. Needs Python 3.9–3.11
# (PyTorch 2.1.2 has no wheels for later versions; override with PYTHON=python3.11) and about 15 GB of free disk for
# the downloads and intermediate models, kept in WORK_DIR (default: a new temporary directory). Runs on CPU.
set -euo pipefail

OUT=${1:?usage: $0 <models dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
WORK=${WORK_DIR:-$(mktemp -d -t djembed-export-XXXXXX)}

# name  task  repository  revision
MODELS=(
  "bge-m3             embed   BAAI/bge-m3              5617a9f61b028005a4858fdac845db406aefb181"
  "bge-reranker-v2-m3 rerank  BAAI/bge-reranker-v2-m3  953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)

version=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case $version in
  3.9|3.10|3.11) ;;
  *) echo "Python $version: PyTorch 2.1.2 needs 3.9–3.11, set PYTHON" >&2; exit 1 ;;
esac

# Two environments with pinned versions: the exporter must be old enough for its attention to fuse, the optimizer
# should be current.
if [ ! -x "$WORK/export-env/bin/python" ]; then
  "$PYTHON" -m venv "$WORK/export-env"
  "$WORK/export-env/bin/pip" install -q --upgrade pip
  "$WORK/export-env/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
  "$WORK/export-env/bin/pip" install -q numpy==1.26.4 transformers==4.44.2 onnx==1.16.2 sentencepiece==0.2.0
fi
if [ ! -x "$WORK/optimize-env/bin/python" ]; then
  "$PYTHON" -m venv "$WORK/optimize-env"
  "$WORK/optimize-env/bin/pip" install -q --upgrade pip
  "$WORK/optimize-env/bin/pip" install -q onnxruntime==1.30.0 onnx==1.23.0 sympy==1.14.0 packaging==25.0
fi
EXPORT_PY="$WORK/export-env/bin/python"
OPTIMIZE_PY="$WORK/optimize-env/bin/python"

for entry in "${MODELS[@]}"; do
  read -r name task repo revision <<< "$entry"
  echo "== $name"
  fp32="$WORK/$name-fp32"
  fp16="$WORK/$name-fp16"
  packed="$OUT/$name-fp16-packed"
  mkdir -p "$fp16" "$packed"

  "$EXPORT_PY" "$HERE/export/export_onnx.py" --task "$task" --repo "$repo" --revision "$revision" --out "$fp32"
  "$OPTIMIZE_PY" "$HERE/export/verify_export.py" "$fp32"

  # XLM-RoBERTa large matches the optimizer's BERT patterns: 16 heads, hidden size 1024.
  "$OPTIMIZE_PY" -m onnxruntime.transformers.optimizer \
      --input "$fp32/model.onnx" --output "$fp16/model.onnx" \
      --model_type bert --num_heads 16 --hidden_size 1024 --opt_level 0 --float16 --use_external_data_format
  "$OPTIMIZE_PY" -m onnxruntime.transformers.convert_to_packing_mode \
      --input "$fp16/model.onnx" --output "$packed/model.onnx" --use_external_data_format

  for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model; do
    if [ -f "$fp32/$file" ]; then cp "$fp32/$file" "$packed/"; fi
  done
  chmod a+r "$packed"/*
  "$OPTIMIZE_PY" "$HERE/export/check_graph.py" "$packed/model.onnx"
done

echo "Done. Intermediate fp32 and fused fp16 (unpacked) models are in $WORK"

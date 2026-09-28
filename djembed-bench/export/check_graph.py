"""Fails unless a model is fully fused and in packing mode, with the inputs and outputs Djembed expects."""
import collections
import sys

import onnx
from onnx import TensorProto

LAYERS = 24

path = sys.argv[1]
model = onnx.load(path, load_external_data=False)
ops = collections.Counter((node.domain + "." if node.domain else "") + node.op_type for node in model.graph.node)
kind = lambda value: TensorProto.DataType.Name(value.type.tensor_type.elem_type)

expected = {
    "com.microsoft.PackedAttention": LAYERS,
    "com.microsoft.SkipLayerNormalization": 2 * LAYERS,
    "com.microsoft.BiasGelu": LAYERS,
    "com.microsoft.RemovePadding": 1,
    "com.microsoft.RestorePadding": 1,
}
problems = [f"{op}: {ops[op]} (expected {count})" for op, count in expected.items() if ops[op] != count]
inputs = {value.name: kind(value) for value in model.graph.input}
if inputs != {"input_ids": "INT64", "attention_mask": "INT64"}:
    problems.append(f"inputs {inputs}")
outputs = {value.name: kind(value) for value in model.graph.output}
if any(dtype != "FLOAT" for dtype in outputs.values()):
    problems.append(f"outputs {outputs}")

print(f"{path}: {dict((op, ops[op]) for op in expected)} outputs={outputs}")
if problems:
    sys.exit("not a fused, packed model: " + "; ".join(problems))

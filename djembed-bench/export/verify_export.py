"""Checks an fp32 ONNX export against the PyTorch outputs saved next to it (reference.npz), on CPU."""
import os
import sys

import numpy as np
import onnxruntime as ort

TOLERANCE = 1e-3

directory = sys.argv[1]
reference = np.load(os.path.join(directory, "reference.npz"))
session = ort.InferenceSession(os.path.join(directory, "model.onnx"), providers=["CPUExecutionProvider"])
feeds = {"input_ids": reference["input_ids"], "attention_mask": reference["attention_mask"]}
names = [output.name for output in session.get_outputs()]

worst = 0.0
for name, value in zip(names, session.run(names, feeds)):
    expected = reference[name]
    if name == "token_embeddings":
        # Padding positions carry no meaning; compare real tokens only.
        mask = feeds["attention_mask"].astype(bool)
        value, expected = value[mask], expected[mask]
    worst = max(worst, float(np.abs(value - expected).max()))

print(f"{os.path.basename(directory)}: max |onnx - torch| = {worst:.2e}")
if worst > TOLERANCE:
    sys.exit(f"export does not match PyTorch (tolerance {TOLERANCE})")

"""Exports a Hugging Face XLM-RoBERTa encoder to fp32 ONNX with classic (eager) attention.

Run with PyTorch 2.1 and transformers 4.44: the attention and mask patterns they emit are the ones ONNX Runtime's
transformer optimizer fuses. Newer exporters (SDPA attention, the transformers 5 mask) leave attention unfused.

Writes the model, its tokenizer and config, and reference.npz: PyTorch's own outputs on sample inputs, which
verify_export.py compares the ONNX model against.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

SAMPLES = [
    ("Qual è il termine di prescrizione ordinario?",
     "Salvi i casi in cui la legge dispone diversamente, i diritti si estinguono per prescrizione con il decorso di dieci anni."),
    ("Qual è il termine di prescrizione ordinario?", "La ricetta prevede farina, uova e zucchero."),
    ("what is a vector database", "A vector database indexes embeddings for similarity search over large collections."),
]


class SentenceEmbedding(torch.nn.Module):
    """bge-m3's sentence-transformers head: CLS pooling, then L2 normalisation."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden, F.normalize(hidden[:, 0], p=2, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["embed", "rerank"], required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    if args.task == "embed":
        encoder = AutoModel.from_pretrained(args.repo, revision=args.revision, attn_implementation="eager",
                                            add_pooling_layer=False)
        model = SentenceEmbedding(encoder).eval()
        config = encoder.config
        inputs = tokenizer([query for query, _ in SAMPLES] + [doc for _, doc in SAMPLES], padding=True, return_tensors="pt")
        outputs = ["token_embeddings", "sentence_embedding"]
        dynamic = {"token_embeddings": {0: "batch_size", 1: "sequence_length"}, "sentence_embedding": {0: "batch_size"}}
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.repo, revision=args.revision,
                                                                   attn_implementation="eager").eval()
        config = model.config
        inputs = tokenizer([list(pair) for pair in SAMPLES], padding=True, return_tensors="pt")
        outputs = ["logits"]
        dynamic = {"logits": {0: "batch_size"}}

    feeds = (inputs["input_ids"], inputs["attention_mask"])
    with torch.no_grad():
        torch.onnx.export(
            model, feeds, os.path.join(args.out, "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=outputs,
            dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                          "attention_mask": {0: "batch_size", 1: "sequence_length"}, **dynamic},
            opset_version=14,
            do_constant_folding=True,
        )
        reference = model(*feeds)
    reference = reference if isinstance(reference, tuple) else (reference.logits,)

    np.savez(os.path.join(args.out, "reference.npz"),
             input_ids=inputs["input_ids"].numpy(), attention_mask=inputs["attention_mask"].numpy(),
             **{name: value.numpy() for name, value in zip(outputs, reference)})
    tokenizer.save_pretrained(args.out)
    config.save_pretrained(args.out)
    print(f"exported {args.repo}@{args.revision[:7]} to {args.out}")


if __name__ == "__main__":
    main()

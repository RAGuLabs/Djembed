package com.ragulabs.djembed.core.internal;

/**
 * What a single forward pass may hold.
 *
 * @param maxRows      most sequences in the pass
 * @param tokenBudget  work allowed per pass: {@code rows × longest} for padded models, the sum of real tokens for
 *                     packed ones, whose encoder never computes padding
 * @param packed       whether the model removes padding inside its encoder
 * @param paddedTokens bound on {@code rows × longest} in any case: the input tensors and the layers around the encoder
 *                     (embeddings, pooling) still see the padded shape. It sizes the workspace.
 */
public record BatchLimits(int maxRows, long tokenBudget, boolean packed, long paddedTokens) {

    /**
     * For packed half-precision models the padded shape only feeds the cheap layers around the encoder, so it may be
     * this many times larger than the budget: batches of mixed lengths stay whole while their memory stays bounded.
     */
    static final int PACKED_PADDING_FACTOR = 4;

    public BatchLimits {
        if (maxRows < 1 || tokenBudget < 1 || paddedTokens < tokenBudget) {
            throw new IllegalArgumentException("invalid batch limits: rows " + maxRows + ", budget " + tokenBudget
                    + ", padded " + paddedTokens);
        }
    }

    /**
     * @param maxInputTokens longest possible sequence: a pass must always be able to hold one
     * @param halfPrecision  whether the model computes in float16/bfloat16. ONNX Runtime's {@code PackedAttention}
     *                       has linear-memory kernels (fused, memory-efficient) for fp16 only; in fp32 it falls back to
     *                       an unfused path whose workspace holds two {@code rows × heads × length²} score matrices of
     *                       the padded shape ({@code packed_attention.cc}, {@code attention_impl.cu}), so a fp32
     *                       packed model keeps the padded bound of an unpacked one
     */
    public static BatchLimits of(int maxRows, int tokenBudget, int maxInputTokens, boolean packed, boolean halfPrecision) {
        long budget = Math.max(tokenBudget, maxInputTokens);
        long padded = packed && halfPrecision ? Math.max(budget * PACKED_PADDING_FACTOR, maxInputTokens) : budget;
        return new BatchLimits(maxRows, budget, packed, padded);
    }

    /** Padding the encoder computes for a pass of {@code rows} rows of {@code rowLength}, {@code tokens} real. */
    public long computedPadding(int rows, int rowLength, long tokens) {
        return packed ? 0 : (long) rows * rowLength - tokens;
    }
}

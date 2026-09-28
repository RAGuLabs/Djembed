package com.ragulabs.djembed.core;

/**
 * Execution settings shared by every engine.
 *
 * @param device          where the model runs
 * @param maxBatchSize    upper bound on the rows of a single forward pass; the token budget is the limit that
 *                        normally applies, this one only sizes per-row buffers
 * @param tokenBudget     upper bound on {@code rows × padded length} of a single forward pass; it sizes the
 *                        off-heap workspace and, with length-sorted packing, lets batches of short inputs carry
 *                        many more rows than batches of long ones at the same peak memory
 * @param maxInputTokens  tokens per sequence, special tokens included; {@link #MODEL_MAX} uses the model's own
 *                        limit, a lower value trades context for memory (attention grows quadratically with it)
 * @param maxQueuedInputs inputs (texts, documents) admitted and not yet completed, across all callers; beyond it
 *                        requests fail with {@link EngineOverloadedException}. A request arriving at an idle engine
 *                        is always admitted.
 * @param tf32            on CUDA devices since Ampere, lets float32 matrix multiplications run on tensor cores in
 *                        TensorFloat-32 (10-bit mantissa): much faster, marginally less precise. ONNX Runtime's
 *                        default, hence on by default; ignored elsewhere.
 */
public record EngineOptions(Device device, int maxBatchSize, int tokenBudget, int maxInputTokens, int maxQueuedInputs,
                            boolean tf32) {

    public static final int MODEL_MAX = 0;

    private static final EngineOptions DEFAULTS = new EngineOptions(Device.cpu(), 1_024, 16_384, MODEL_MAX, 8_192, true);

    public EngineOptions {
        if (device == null) {
            throw new IllegalArgumentException("device is required");
        }
        if (maxBatchSize < 1) {
            throw new IllegalArgumentException("maxBatchSize must be >= 1: " + maxBatchSize);
        }
        if (tokenBudget < 1) {
            throw new IllegalArgumentException("tokenBudget must be >= 1: " + tokenBudget);
        }
        if (maxInputTokens < 0) {
            throw new IllegalArgumentException("maxInputTokens must be >= 0: " + maxInputTokens);
        }
        if (maxQueuedInputs < 1) {
            throw new IllegalArgumentException("maxQueuedInputs must be >= 1: " + maxQueuedInputs);
        }
    }

    public static EngineOptions defaults() {
        return DEFAULTS;
    }

    public EngineOptions withDevice(Device device) {
        return new EngineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, tf32);
    }

    public EngineOptions withMaxBatchSize(int maxBatchSize) {
        return new EngineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, tf32);
    }

    public EngineOptions withTokenBudget(int tokenBudget) {
        return new EngineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, tf32);
    }

    public EngineOptions withMaxInputTokens(int maxInputTokens) {
        return new EngineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, tf32);
    }

    public EngineOptions withMaxQueuedInputs(int maxQueuedInputs) {
        return new EngineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, tf32);
    }

    public EngineOptions withTf32(boolean tf32) {
        return new EngineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, tf32);
    }
}

package com.ragulabs.djembed.core;

/**
 * @param engine    execution settings
 * @param longInput what to do with inputs over the token limit
 * @param pooling   overrides the pooling declared by the model directory; {@code null} keeps the declared one.
 *                  Ignored when the ONNX graph already outputs a pooled {@code sentence_embedding}.
 * @param normalize L2-normalize the returned vectors
 */
public record EmbeddingOptions(EngineOptions engine, LongInputStrategy longInput, PoolingMode pooling, boolean normalize) {

    private static final EmbeddingOptions DEFAULTS =
            new EmbeddingOptions(EngineOptions.defaults(), LongInputStrategy.TRUNCATE, null, true);

    public EmbeddingOptions {
        if (engine == null) {
            throw new IllegalArgumentException("engine options are required");
        }
        if (longInput == null) {
            throw new IllegalArgumentException("longInput is required");
        }
    }

    public static EmbeddingOptions defaults() {
        return DEFAULTS;
    }

    public EmbeddingOptions withEngine(EngineOptions engine) {
        return new EmbeddingOptions(engine, longInput, pooling, normalize);
    }

    public EmbeddingOptions withLongInput(LongInputStrategy longInput) {
        return new EmbeddingOptions(engine, longInput, pooling, normalize);
    }

    public EmbeddingOptions withPooling(PoolingMode pooling) {
        return new EmbeddingOptions(engine, longInput, pooling, normalize);
    }

    public EmbeddingOptions withNormalize(boolean normalize) {
        return new EmbeddingOptions(engine, longInput, pooling, normalize);
    }
}

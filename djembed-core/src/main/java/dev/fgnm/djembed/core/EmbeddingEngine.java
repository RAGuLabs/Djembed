package dev.fgnm.djembed.core;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Turns texts into vectors. Implementations are thread-safe.
 */
public interface EmbeddingEngine extends AutoCloseable {

    /**
     * One vector per text, in input order. Cancelling the future drops the work not yet started.
     *
     * @throws EngineOverloadedException when the engine's queue is full
     */
    CompletableFuture<Embeddings> embedAsync(List<String> texts);

    /** Blocking form of {@link #embedAsync}. */
    default Embeddings embed(List<String> texts) {
        return Futures.await(embedAsync(texts));
    }

    int dimension();

    /** Token limit per sequence, special tokens included. */
    int maxInputTokens();

    /** Inputs accepted and not yet completed, across all callers. */
    default int queuedInputs() {
        return 0;
    }

    @Override
    void close();
}

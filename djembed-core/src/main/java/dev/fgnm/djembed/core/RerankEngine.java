package dev.fgnm.djembed.core;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Scores how relevant each document is to a query. Implementations are thread-safe.
 */
public interface RerankEngine extends AutoCloseable {

    /**
     * One score per document, in input order. Cancelling the future drops the work not yet started.
     *
     * @throws EngineOverloadedException when the engine's queue is full
     */
    CompletableFuture<RerankScores> scoreAsync(String query, List<String> documents);

    /** Blocking form of {@link #scoreAsync}. */
    default RerankScores score(String query, List<String> documents) {
        return Futures.await(scoreAsync(query, documents));
    }

    /** Token limit per (query, document) pair, special tokens included. */
    int maxInputTokens();

    /** Inputs accepted and not yet completed, across all callers. */
    default int queuedInputs() {
        return 0;
    }

    @Override
    void close();
}

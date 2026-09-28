package com.ragulabs.djembed.bench;

import java.util.Locale;

/**
 * What a request carries. Every workload is generated from the seed alone, so all targets receive byte-identical
 * texts.
 */
enum Workload {

    /** One short text per request: embedding a search query, where latency matters. */
    QUERY(1, 6, 16),
    /** A batch of passages per request: document ingestion, where throughput matters. */
    INGEST(32, 50, 380),
    /** A query and a batch of candidate documents: the reranking step of retrieval. */
    RERANK(32, 30, 230);

    final int inputs;
    final int minWords;
    final int maxWords;

    Workload(int inputs, int minWords, int maxWords) {
        this.inputs = inputs;
        this.minWords = minWords;
        this.maxWords = maxWords;
    }

    boolean isEmbedding() {
        return this != RERANK;
    }

    String label() {
        return name().toLowerCase(Locale.ROOT);
    }

    static Workload parse(String value) {
        return valueOf(value.trim().toUpperCase(Locale.ROOT));
    }
}

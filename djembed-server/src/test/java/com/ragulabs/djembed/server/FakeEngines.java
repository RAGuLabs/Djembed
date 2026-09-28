package com.ragulabs.djembed.server;

import com.ragulabs.djembed.core.EmbeddingEngine;
import com.ragulabs.djembed.core.Embeddings;
import com.ragulabs.djembed.core.EngineOverloadedException;
import com.ragulabs.djembed.core.RerankEngine;
import com.ragulabs.djembed.core.RerankScores;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Deterministic engines for exercising the HTTP layer without a model.
 */
public final class FakeEngines {

    public static final String EMBED = "fake-embed";
    public static final String RERANK = "fake-rerank";
    public static final int LIMIT = 20;
    /** A text or query that makes the fake engines report a full queue. */
    public static final String OVERLOAD = "overload";

    private FakeEngines() {
    }

    public static ModelRegistry registry() {
        return new ModelRegistry(Map.of(EMBED, new Embedder()), Map.of(RERANK, new Reranker()));
    }

    /** Vector of a text: {@code [length, 1, 0.5]}. Texts longer than {@link #LIMIT} chars count as truncated. */
    static final class Embedder implements EmbeddingEngine {

        @Override
        public CompletableFuture<Embeddings> embedAsync(List<String> texts) {
            if (texts.contains(OVERLOAD)) {
                throw new EngineOverloadedException("queue full");
            }
            return CompletableFuture.supplyAsync(() -> compute(texts));
        }

        private static Embeddings compute(List<String> texts) {
            float[] data = new float[texts.size() * 3];
            long tokens = 0;
            int firstTruncated = -1;
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                data[i * 3] = text.length();
                data[i * 3 + 1] = 1f;
                data[i * 3 + 2] = 0.5f;
                tokens += Math.min(text.length(), LIMIT);
                if (firstTruncated < 0 && text.length() > LIMIT) {
                    firstTruncated = i;
                }
            }
            return new Embeddings(data, texts.size(), 3, tokens, firstTruncated);
        }

        @Override
        public int dimension() {
            return 3;
        }

        @Override
        public int maxInputTokens() {
            return LIMIT;
        }

        @Override
        public void close() {
        }
    }

    /** Score of a document: {@code 0.9} when it contains the query, else its length / 1000. */
    static final class Reranker implements RerankEngine {

        @Override
        public CompletableFuture<RerankScores> scoreAsync(String query, List<String> documents) {
            if (query.equals(OVERLOAD)) {
                return CompletableFuture.failedFuture(new EngineOverloadedException("queue full"));
            }
            return CompletableFuture.supplyAsync(() -> compute(query, documents));
        }

        private static RerankScores compute(String query, List<String> documents) {
            float[] scores = new float[documents.size()];
            for (int i = 0; i < scores.length; i++) {
                String document = documents.get(i);
                scores[i] = document.contains(query) ? 0.9f : document.length() / 1000f;
            }
            return new RerankScores(scores, documents.size() * 10L);
        }

        @Override
        public int maxInputTokens() {
            return LIMIT;
        }

        @Override
        public void close() {
        }
    }
}

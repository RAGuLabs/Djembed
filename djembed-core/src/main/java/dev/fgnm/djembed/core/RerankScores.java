package dev.fgnm.djembed.core;

import java.util.Objects;

/**
 * Relevance scores of one {@link RerankEngine#score} call, in the order the documents were given.
 */
public final class RerankScores {

    private final float[] scores;
    private final long promptTokens;

    public RerankScores(float[] scores, long promptTokens) {
        this.scores = Objects.requireNonNull(scores, "scores");
        this.promptTokens = promptTokens;
    }

    public int count() {
        return scores.length;
    }

    public float score(int index) {
        return scores[index];
    }

    /** Tokens fed to the model, special tokens included. */
    public long promptTokens() {
        return promptTokens;
    }
}

package dev.fgnm.djembed.core;

/**
 * Transformation applied to a reranker logit.
 */
public enum ScoreActivation {
    /** Relevance in {@code [0, 1]}, comparable across queries. */
    SIGMOID,
    /** The raw logit. */
    NONE
}

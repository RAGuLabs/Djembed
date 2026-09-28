package com.ragulabs.djembed.core;

/**
 * How token vectors collapse into one sentence vector, for models whose graph does not already do it.
 */
public enum PoolingMode {
    /** First token ({@code [CLS]} / {@code <s>}), e.g. BGE. */
    CLS,
    /** Mean of the non-padding tokens, e.g. E5, MiniLM. */
    MEAN,
    /** Last non-padding token, e.g. decoder-based embedders such as Qwen3-Embedding. */
    LAST_TOKEN
}

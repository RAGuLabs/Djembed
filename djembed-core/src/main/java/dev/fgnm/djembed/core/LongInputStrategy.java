package dev.fgnm.djembed.core;

/**
 * What happens to an input longer than the engine's token limit.
 */
public enum LongInputStrategy {
    /** Keep the leading tokens and drop the rest, like Cohere, OpenAI and TEI. */
    TRUNCATE,
    /**
     * Split at word boundaries into windows that each fit the limit, embed every window and return their
     * token-weighted average, so the whole text contributes to the vector.
     */
    CHUNK
}

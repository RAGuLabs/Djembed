package com.ragulabs.djembed.core;

/**
 * Receives what an engine does, for metrics. Called on the engine's device thread: implementations must be cheap
 * and must not block.
 */
public interface EngineObserver {

    EngineObserver NONE = new EngineObserver() {
    };

    /**
     * A forward pass completed.
     *
     * @param rows      sequences in the batch
     * @param rowLength padded length of every row
     * @param tokens    real tokens; {@code rows × rowLength − tokens} were padding
     * @param nanos     time spent in the forward pass, input transfer and output copy included
     */
    default void forwardPass(int rows, int rowLength, long tokens, long nanos) {
    }
}

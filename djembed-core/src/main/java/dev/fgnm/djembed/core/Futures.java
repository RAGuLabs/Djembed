package dev.fgnm.djembed.core;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

final class Futures {

    private Futures() {
    }

    /** Waits for {@code future}, rethrowing its failure as thrown rather than wrapped. */
    static <T> T await(CompletableFuture<T> future) {
        try {
            return future.join();
        } catch (CompletionException e) {
            if (e.getCause() instanceof RuntimeException cause) {
                throw cause;
            }
            if (e.getCause() instanceof Error cause) {
                throw cause;
            }
            throw e;
        }
    }
}

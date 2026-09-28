package com.ragulabs.djembed.core;

import java.io.Serial;

/**
 * An engine refused work because its queue is full; retrying later may succeed.
 */
public class EngineOverloadedException extends DjembedException {

    @Serial
    private static final long serialVersionUID = 1L;

    public EngineOverloadedException(String message) {
        super(message);
    }
}

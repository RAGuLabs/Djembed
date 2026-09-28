package dev.fgnm.djembed.core;

import java.io.Serial;

public class DjembedException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 1L;

    public DjembedException(String message) {
        super(message);
    }

    public DjembedException(String message, Throwable cause) {
        super(message, cause);
    }
}

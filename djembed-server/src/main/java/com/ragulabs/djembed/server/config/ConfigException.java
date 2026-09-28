package com.ragulabs.djembed.server.config;

import java.io.Serial;

public class ConfigException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 1L;

    public ConfigException(String message, Throwable cause) {
        super(message, cause);
    }
}

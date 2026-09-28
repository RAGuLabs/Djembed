package dev.fgnm.djembed.server.api;

import com.linecorp.armeria.common.HttpStatus;

import java.io.Serial;

/**
 * A request the API refuses, answered with {@code status} and {@code message}, and for APIs that have them, a
 * machine-readable {@code code}.
 */
public final class ApiException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 1L;

    private final transient HttpStatus status;
    private final String code;

    public ApiException(HttpStatus status, String message) {
        this(status, message, null);
    }

    public ApiException(HttpStatus status, String message, String code) {
        super(message, null, false, false);
        this.status = status;
        this.code = code;
    }

    public static ApiException badRequest(String message) {
        return new ApiException(HttpStatus.BAD_REQUEST, message);
    }

    public HttpStatus status() {
        return status;
    }

    /** Machine-readable error code, or {@code null}. */
    public String code() {
        return code;
    }
}

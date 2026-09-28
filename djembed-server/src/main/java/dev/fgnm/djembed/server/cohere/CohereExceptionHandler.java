package dev.fgnm.djembed.server.cohere;

import com.linecorp.armeria.common.HttpRequest;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.common.util.Exceptions;
import com.linecorp.armeria.server.ServiceRequestContext;
import com.linecorp.armeria.server.annotation.ExceptionHandlerFunction;
import dev.fgnm.djembed.core.EngineOverloadedException;
import dev.fgnm.djembed.server.api.ApiException;
import dev.fgnm.djembed.server.json.JsonResponses;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Renders failures as Cohere does: {@code {"id": ..., "message": ...}} with the matching status.
 */
public final class CohereExceptionHandler implements ExceptionHandlerFunction {

    private static final Logger log = LoggerFactory.getLogger(CohereExceptionHandler.class);

    @Override
    public HttpResponse handleException(ServiceRequestContext ctx, HttpRequest req, Throwable thrown) {
        Throwable cause = Exceptions.peel(thrown);
        return switch (cause) {
            case ApiException e -> error(ctx, e.status(), e.getMessage());
            case EngineOverloadedException e -> error(ctx, HttpStatus.SERVICE_UNAVAILABLE, e.getMessage());
            case IllegalArgumentException e -> error(ctx, HttpStatus.BAD_REQUEST, e.getMessage());
            // Engines refuse work once closed, which only happens while the server shuts down.
            case IllegalStateException e -> error(ctx, HttpStatus.SERVICE_UNAVAILABLE, e.getMessage());
            default -> {
                log.error("Request {} {} failed", req.method(), req.path(), cause);
                yield error(ctx, HttpStatus.INTERNAL_SERVER_ERROR, "internal error");
            }
        };
    }

    /** Unauthorised requests, in the same format. */
    public static HttpResponse unauthorized(ServiceRequestContext ctx, String message) {
        return error(ctx, HttpStatus.UNAUTHORIZED, message);
    }

    static HttpResponse error(ServiceRequestContext ctx, HttpStatus status, String message) {
        return JsonResponses.of(ctx, status, 256, json -> {
            json.writeStartObject();
            json.writeStringField("id", ctx.id().text());
            json.writeStringField("message", message);
            json.writeEndObject();
        });
    }
}

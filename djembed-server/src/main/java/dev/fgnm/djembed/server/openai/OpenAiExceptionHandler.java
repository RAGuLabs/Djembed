package dev.fgnm.djembed.server.openai;

import com.linecorp.armeria.common.HttpRequest;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.common.HttpStatusClass;
import com.linecorp.armeria.common.util.Exceptions;
import com.linecorp.armeria.server.ServiceRequestContext;
import com.linecorp.armeria.server.annotation.ExceptionHandlerFunction;
import dev.fgnm.djembed.core.EngineOverloadedException;
import dev.fgnm.djembed.server.api.ApiException;
import dev.fgnm.djembed.server.json.JsonResponses;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Renders failures as OpenAI does: {@code {"error": {"message", "type", "param", "code"}}}.
 */
public final class OpenAiExceptionHandler implements ExceptionHandlerFunction {

    private static final Logger log = LoggerFactory.getLogger(OpenAiExceptionHandler.class);

    private static final String INVALID_REQUEST = "invalid_request_error";
    private static final String SERVER_ERROR = "server_error";

    @Override
    public HttpResponse handleException(ServiceRequestContext ctx, HttpRequest req, Throwable thrown) {
        Throwable cause = Exceptions.peel(thrown);
        return switch (cause) {
            case ApiException e -> error(ctx, e.status(), e.getMessage(), e.code());
            case EngineOverloadedException e -> error(ctx, HttpStatus.SERVICE_UNAVAILABLE, e.getMessage(), null);
            case IllegalArgumentException e -> error(ctx, HttpStatus.BAD_REQUEST, e.getMessage(), null);
            // Engines refuse work once closed, which only happens while the server shuts down.
            case IllegalStateException e -> error(ctx, HttpStatus.SERVICE_UNAVAILABLE, e.getMessage(), null);
            default -> {
                log.error("Request {} {} failed", req.method(), req.path(), cause);
                yield error(ctx, HttpStatus.INTERNAL_SERVER_ERROR, "internal error", null);
            }
        };
    }

    /** Unauthorised requests, in the same format. */
    public static HttpResponse unauthorized(ServiceRequestContext ctx, String message) {
        return error(ctx, HttpStatus.UNAUTHORIZED, message, "invalid_api_key");
    }

    static HttpResponse error(ServiceRequestContext ctx, HttpStatus status, String message, String code) {
        String type = status.codeClass() == HttpStatusClass.SERVER_ERROR ? SERVER_ERROR : INVALID_REQUEST;
        return JsonResponses.of(ctx, status, 256, json -> {
            json.writeStartObject();
            json.writeObjectFieldStart("error");
            json.writeStringField("message", message);
            json.writeStringField("type", type);
            json.writeNullField("param");
            if (code != null) {
                json.writeStringField("code", code);
            } else {
                json.writeNullField("code");
            }
            json.writeEndObject();
            json.writeEndObject();
        });
    }
}

package dev.fgnm.djembed.server.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.linecorp.armeria.common.AggregatedHttpRequest;

import java.io.IOException;
import java.io.UncheckedIOException;

public final class JsonBodies {

    private JsonBodies() {
    }

    /** Reads the request body as {@code type}; malformed JSON becomes a 400. */
    public static <T> T parse(ObjectMapper mapper, AggregatedHttpRequest http, Class<T> type) {
        try {
            return mapper.readValue(http.content().array(), type);
        } catch (JsonProcessingException e) {
            throw ApiException.badRequest("invalid request body: " + e.getOriginalMessage());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}

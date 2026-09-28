package dev.fgnm.djembed.server.json;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.StreamWriteFeature;
import com.linecorp.armeria.common.HttpData;
import com.linecorp.armeria.common.HttpHeaderNames;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.common.MediaType;
import com.linecorp.armeria.common.ResponseHeaders;
import com.linecorp.armeria.server.ServiceRequestContext;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufOutputStream;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;

/**
 * Writes JSON responses straight into a pooled Netty buffer that Armeria sends and releases, so a response body is
 * never materialised as a heap {@code byte[]} or {@code String}.
 */
public final class JsonResponses {

    private static final JsonFactory FACTORY = JsonFactory.builder()
            .disable(StreamWriteFeature.AUTO_CLOSE_TARGET)
            .build();

    private JsonResponses() {
    }

    @FunctionalInterface
    public interface Body {
        void write(JsonGenerator json) throws IOException;
    }

    /**
     * @param sizeHint expected body size in bytes, so large bodies are allocated once instead of grown
     */
    public static HttpResponse of(ServiceRequestContext ctx, HttpStatus status, int sizeHint, Body body) {
        ByteBuf buffer = ctx.alloc().buffer(Math.max(256, sizeHint));
        try (OutputStream out = new ByteBufOutputStream(buffer);
             JsonGenerator json = FACTORY.createGenerator(out)) {
            body.write(json);
        } catch (IOException e) {
            buffer.release();
            throw new UncheckedIOException(e);
        } catch (RuntimeException | Error e) {
            buffer.release();
            throw e;
        }
        return HttpResponse.of(ResponseHeaders.of(status, HttpHeaderNames.CONTENT_TYPE,
                MediaType.JSON_UTF_8), HttpData.wrap(buffer));
    }
}

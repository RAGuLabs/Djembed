package com.ragulabs.djembed.server.openai;

import com.fasterxml.jackson.core.Base64Variants;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.linecorp.armeria.common.AggregatedHttpRequest;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.server.ServiceRequestContext;
import com.linecorp.armeria.server.annotation.ExceptionHandler;
import com.linecorp.armeria.server.annotation.Get;
import com.linecorp.armeria.server.annotation.Post;
import com.ragulabs.djembed.core.CpuPool;
import com.ragulabs.djembed.core.EmbeddingEngine;
import com.ragulabs.djembed.core.Embeddings;
import com.ragulabs.djembed.server.ModelRegistry;
import com.ragulabs.djembed.server.api.ApiException;
import com.ragulabs.djembed.server.api.JsonBodies;
import com.ragulabs.djembed.server.json.FloatWriter;
import com.ragulabs.djembed.server.json.JsonResponses;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * OpenAI-compatible {@code /v1/embeddings} and {@code /v1/models}.
 *
 * <p>As with OpenAI, an input over the model's token limit is refused rather than truncated (unless the model is
 * configured to chunk long inputs), and {@code dimensions} must match what the model produces.
 */
@ExceptionHandler(OpenAiExceptionHandler.class)
public final class OpenAiService {

    private static final ObjectMapper JSON = JsonMapper.builder()
            .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .build();

    private static final String FLOAT = "float";
    private static final String BASE64 = "base64";
    private static final String OWNER = "djembed";

    private final ModelRegistry registry;
    private final long created = Instant.now().getEpochSecond();

    public OpenAiService(ModelRegistry registry) {
        this.registry = registry;
    }

    private record Call(String model, EmbeddingEngine engine, List<String> texts, boolean base64) {
    }

    @Post("/v1/embeddings")
    public CompletableFuture<HttpResponse> embeddings(ServiceRequestContext ctx, AggregatedHttpRequest http) {
        return CompletableFuture.supplyAsync(() -> prepare(http), CpuPool.executor())
                .thenCompose(call -> {
                    CompletableFuture<Embeddings> work = call.engine().embedAsync(call.texts());
                    ctx.whenRequestCancelling().thenRun(() -> work.cancel(false));
                    return work.thenApply(embeddings -> render(ctx, call, embeddings));
                });
    }

    @Get("/v1/models")
    public HttpResponse models(ServiceRequestContext ctx) {
        return JsonResponses.of(ctx, HttpStatus.OK, 512, json -> {
            json.writeStartObject();
            json.writeStringField("object", "list");
            json.writeArrayFieldStart("data");
            for (String name : registry.names()) {
                json.writeStartObject();
                json.writeStringField("id", name);
                json.writeStringField("object", "model");
                json.writeNumberField("created", created);
                json.writeStringField("owned_by", OWNER);
                json.writeEndObject();
            }
            json.writeEndArray();
            json.writeEndObject();
        });
    }

    private Call prepare(AggregatedHttpRequest http) {
        EmbeddingsRequest request = JsonBodies.parse(JSON, http, EmbeddingsRequest.class);
        String model = request.model();
        if (model == null || model.isBlank()) {
            throw ApiException.badRequest("model is required");
        }
        EmbeddingEngine engine = registry.embedder(model);
        if (engine == null) {
            if (registry.contains(model)) {
                throw ApiException.badRequest("model '" + model + "' does not support embeddings");
            }
            throw new ApiException(HttpStatus.NOT_FOUND, "The model '" + model + "' does not exist", "model_not_found");
        }

        List<String> texts = request.input();
        if (texts == null || texts.isEmpty()) {
            throw ApiException.badRequest("input must not be empty");
        }
        for (int i = 0; i < texts.size(); i++) {
            if (texts.get(i) == null || texts.get(i).isEmpty()) {
                throw ApiException.badRequest("input[" + i + "] must be a non-empty string");
            }
        }
        String format = request.encodingFormat() == null ? FLOAT : request.encodingFormat();
        if (!format.equals(FLOAT) && !format.equals(BASE64)) {
            throw ApiException.badRequest("encoding_format must be float or base64, got '" + format + "'");
        }
        if (request.dimensions() != null && request.dimensions() != engine.dimension()) {
            throw ApiException.badRequest("dimensions " + request.dimensions() + " is not available: model '" + model
                    + "' produces " + engine.dimension() + " dimensions");
        }
        return new Call(model, engine, texts, format.equals(BASE64));
    }

    private static HttpResponse render(ServiceRequestContext ctx, Call call, Embeddings embeddings) {
        if (embeddings.firstTruncated() >= 0) {
            throw ApiException.badRequest("input[" + embeddings.firstTruncated() + "] exceeds the maximum context length of "
                    + call.engine().maxInputTokens() + " tokens");
        }
        int dimension = embeddings.dimension();
        long sizeHint = 256L + (long) embeddings.count() * (dimension * (call.base64() ? 6L : 12L) + 48);
        return JsonResponses.of(ctx, HttpStatus.OK, (int) Math.min(sizeHint, Integer.MAX_VALUE - 8), json -> {
            json.writeStartObject();
            json.writeStringField("object", "list");
            json.writeArrayFieldStart("data");
            if (call.base64()) {
                writeBase64(json, embeddings);
            } else {
                writeFloats(json, embeddings);
            }
            json.writeEndArray();
            json.writeStringField("model", call.model());
            json.writeObjectFieldStart("usage");
            json.writeNumberField("prompt_tokens", embeddings.promptTokens());
            json.writeNumberField("total_tokens", embeddings.promptTokens());
            json.writeEndObject();
            json.writeEndObject();
        });
    }

    private static void writeFloats(JsonGenerator json, Embeddings embeddings) throws IOException {
        FloatWriter floats = new FloatWriter();
        float[] row = new float[embeddings.dimension()];
        for (int i = 0; i < embeddings.count(); i++) {
            embeddings.copyTo(i, row, 0);
            startItem(json, i);
            json.writeStartArray();
            for (float value : row) {
                floats.write(json, value);
            }
            json.writeEndArray();
            json.writeEndObject();
        }
    }

    /** Little-endian float32 bytes, base64-encoded by the generator straight into the output from one reused buffer. */
    private static void writeBase64(JsonGenerator json, Embeddings embeddings) throws IOException {
        int dimension = embeddings.dimension();
        byte[] bytes = new byte[dimension * Float.BYTES];
        ByteBuffer view = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < embeddings.count(); i++) {
            for (int j = 0; j < dimension; j++) {
                view.putFloat(j * Float.BYTES, embeddings.get(i, j));
            }
            startItem(json, i);
            json.writeBinary(Base64Variants.MIME_NO_LINEFEEDS, bytes, 0, bytes.length);
            json.writeEndObject();
        }
    }

    private static void startItem(JsonGenerator json, int index) throws IOException {
        json.writeStartObject();
        json.writeStringField("object", "embedding");
        json.writeNumberField("index", index);
        json.writeFieldName("embedding");
    }
}

package com.ragulabs.djembed.server.cohere;

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
import com.linecorp.armeria.server.annotation.Post;
import com.ragulabs.djembed.core.CpuPool;
import com.ragulabs.djembed.core.EmbeddingEngine;
import com.ragulabs.djembed.core.Embeddings;
import com.ragulabs.djembed.core.RerankEngine;
import com.ragulabs.djembed.core.RerankScores;
import com.ragulabs.djembed.server.ModelRegistry;
import com.ragulabs.djembed.server.api.ApiException;
import com.ragulabs.djembed.server.api.JsonBodies;
import com.ragulabs.djembed.server.json.FloatWriter;
import com.ragulabs.djembed.server.json.JsonResponses;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * Cohere-compatible embed and rerank endpoints, API versions 1 and 2.
 *
 * <p>Fully asynchronous: the body is parsed and validated on the CPU pool (never on an event loop), the
 * engine batches the work with other requests, and the response is rendered on the thread that completes it. No
 * thread waits for the device. A client that goes away cancels its work if it has not reached the device yet.
 *
 * <p>Text only. Request fields that would change the result in ways the served model cannot honour (other
 * embedding types, image inputs, start truncation, token caps below the model limit, rank fields) are refused with
 * a 400 rather than silently ignored. Scheduling hints ({@code priority}) and chunking hints ({@code max_chunks_per_doc})
 * are accepted and ignored.
 */
@ExceptionHandler(CohereExceptionHandler.class)
public final class CohereService {

    private static final ObjectMapper JSON = JsonMapper.builder()
            .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .build();

    private static final Set<String> TEXT_INPUT_TYPES = Set.of("search_document", "search_query", "classification", "clustering");
    private static final String FLOAT = "float";

    private enum Version {
        V1("1"), V2("2");

        final String number;

        Version(String number) {
            this.number = number;
        }
    }

    private final ModelRegistry registry;

    public CohereService(ModelRegistry registry) {
        this.registry = registry;
    }

    @Post("/v1/embed")
    public CompletableFuture<HttpResponse> embedV1(ServiceRequestContext ctx, AggregatedHttpRequest request) {
        return embed(ctx, request, Version.V1);
    }

    @Post("/v2/embed")
    public CompletableFuture<HttpResponse> embedV2(ServiceRequestContext ctx, AggregatedHttpRequest request) {
        return embed(ctx, request, Version.V2);
    }

    @Post("/v1/rerank")
    public CompletableFuture<HttpResponse> rerankV1(ServiceRequestContext ctx, AggregatedHttpRequest request) {
        return rerank(ctx, request, Version.V1);
    }

    @Post("/v2/rerank")
    public CompletableFuture<HttpResponse> rerankV2(ServiceRequestContext ctx, AggregatedHttpRequest request) {
        return rerank(ctx, request, Version.V2);
    }

    private record EmbedCall(EmbeddingEngine engine, List<String> texts, boolean byType, boolean rejectTruncation) {
    }

    private record RerankCall(RerankEngine engine, String query, List<String> texts, int topN, boolean returnDocuments) {
    }

    private CompletableFuture<HttpResponse> embed(ServiceRequestContext ctx, AggregatedHttpRequest http, Version version) {
        return CompletableFuture.supplyAsync(() -> prepareEmbed(http, version), CpuPool.executor())
                .thenCompose(call -> cancelWith(ctx, call.engine().embedAsync(call.texts()))
                        .thenApply(embeddings -> renderEmbed(ctx, version, call, embeddings)));
    }

    private CompletableFuture<HttpResponse> rerank(ServiceRequestContext ctx, AggregatedHttpRequest http, Version version) {
        return CompletableFuture.supplyAsync(() -> prepareRerank(http, version), CpuPool.executor())
                .thenCompose(call -> cancelWith(ctx, call.engine().scoreAsync(call.query(), call.texts()))
                        .thenApply(scores -> renderRerank(ctx, version, call, scores)));
    }

    /** Cancels engine work when the client goes away, so it never reaches the device if it has not already. */
    private static <T> CompletableFuture<T> cancelWith(ServiceRequestContext ctx, CompletableFuture<T> work) {
        ctx.whenRequestCancelling().thenRun(() -> work.cancel(false));
        return work;
    }

    private EmbedCall prepareEmbed(AggregatedHttpRequest http, Version version) {
        EmbedRequest request = JsonBodies.parse(JSON, http, EmbedRequest.class);

        String model = modelName(request.model(), version, registry.soleEmbedder());
        EmbeddingEngine engine = registry.embedder(model);
        if (engine == null) {
            throw unknownModel(model, "embed");
        }

        List<String> texts = texts(request, version);
        validateInputType(request.inputType());
        boolean byType = embeddingTypes(request.embeddingTypes());
        if (request.outputDimension() != null && request.outputDimension() != engine.dimension()) {
            throw ApiException.badRequest("output_dimension " + request.outputDimension() + " is not available: model '"
                    + model + "' produces " + engine.dimension() + " dimensions");
        }
        if (request.maxTokens() != null && request.maxTokens() < engine.maxInputTokens()) {
            throw ApiException.badRequest("max_tokens below the model limit (" + engine.maxInputTokens() + ") is not supported");
        }
        return new EmbedCall(engine, texts, byType, truncateNone(request.truncate()));
    }

    private static HttpResponse renderEmbed(ServiceRequestContext ctx, Version version, EmbedCall call, Embeddings embeddings) {
        if (call.rejectTruncation() && embeddings.firstTruncated() >= 0) {
            throw ApiException.badRequest("texts[" + embeddings.firstTruncated() + "] exceeds the model limit of "
                    + call.engine().maxInputTokens() + " tokens and truncate is NONE");
        }
        List<String> texts = call.texts();
        boolean byType = call.byType();

        long sizeHint = 256L + (long) embeddings.count() * embeddings.dimension() * 12;
        for (String text : texts) {
            sizeHint += text.length() + 4;
        }
        return JsonResponses.of(ctx, HttpStatus.OK, (int) Math.min(sizeHint, Integer.MAX_VALUE - 8), json -> {
            json.writeStartObject();
            json.writeStringField("id", ctx.id().text());
            if (version == Version.V1 && !byType) {
                json.writeStringField("response_type", "embeddings_floats");
                json.writeFieldName("embeddings");
                writeMatrix(json, embeddings);
            } else {
                if (version == Version.V1) {
                    json.writeStringField("response_type", "embeddings_by_type");
                }
                json.writeObjectFieldStart("embeddings");
                json.writeFieldName(FLOAT);
                writeMatrix(json, embeddings);
                json.writeEndObject();
            }
            json.writeArrayFieldStart("texts");
            for (String text : texts) {
                json.writeString(text);
            }
            json.writeEndArray();
            writeMeta(json, version, embeddings.promptTokens(), "input_tokens", embeddings.promptTokens());
            json.writeEndObject();
        });
    }

    private RerankCall prepareRerank(AggregatedHttpRequest http, Version version) {
        RerankRequest request = JsonBodies.parse(JSON, http, RerankRequest.class);

        String model = modelName(request.model(), version, registry.soleReranker());
        RerankEngine engine = registry.reranker(model);
        if (engine == null) {
            throw unknownModel(model, "rerank");
        }

        if (request.query() == null) {
            throw ApiException.badRequest("query is required");
        }
        List<RerankRequest.Document> documents = request.documents();
        if (documents == null || documents.isEmpty()) {
            throw ApiException.badRequest("documents must not be empty");
        }
        List<String> texts = new ArrayList<>(documents.size());
        for (int i = 0; i < documents.size(); i++) {
            RerankRequest.Document document = documents.get(i);
            if (document == null || document.text() == null) {
                throw ApiException.badRequest("documents[" + i + "] has no text");
            }
            texts.add(document.text());
        }
        if (request.rankFields() != null && !request.rankFields().isEmpty()) {
            throw ApiException.badRequest("rank_fields is not supported: pass the text to rank as a string or in a text field");
        }
        if (request.maxTokensPerDoc() != null && request.maxTokensPerDoc() < engine.maxInputTokens()) {
            throw ApiException.badRequest("max_tokens_per_doc below the model limit (" + engine.maxInputTokens() + ") is not supported");
        }
        int topN = texts.size();
        if (request.topN() != null) {
            if (request.topN() < 1) {
                throw ApiException.badRequest("top_n must be at least 1");
            }
            topN = Math.min(request.topN(), texts.size());
        }
        return new RerankCall(engine, request.query(), texts, topN, Boolean.TRUE.equals(request.returnDocuments()));
    }

    private static HttpResponse renderRerank(ServiceRequestContext ctx, Version version, RerankCall call, RerankScores scores) {
        List<String> texts = call.texts();
        boolean returnDocuments = call.returnDocuments();
        int[] ranking = Ranking.descending(scores);

        int results = call.topN();
        long sizeHint = 256L + results * 48L;
        if (returnDocuments) {
            for (int r = 0; r < results; r++) {
                sizeHint += texts.get(ranking[r]).length() + 24;
            }
        }
        return JsonResponses.of(ctx, HttpStatus.OK, (int) Math.min(sizeHint, Integer.MAX_VALUE - 8), json -> {
            FloatWriter floats = new FloatWriter();
            json.writeStartObject();
            json.writeStringField("id", ctx.id().text());
            json.writeArrayFieldStart("results");
            for (int r = 0; r < results; r++) {
                int index = ranking[r];
                json.writeStartObject();
                json.writeNumberField("index", index);
                json.writeFieldName("relevance_score");
                floats.write(json, scores.score(index));
                if (returnDocuments) {
                    json.writeObjectFieldStart("document");
                    json.writeStringField("text", texts.get(index));
                    json.writeEndObject();
                }
                json.writeEndObject();
            }
            json.writeEndArray();
            writeMeta(json, version, scores.promptTokens(), "search_units", searchUnits(texts.size()));
            json.writeEndObject();
        });
    }

    /** The model a request targets: v2 names it explicitly, v1 may leave it out when only one model fits. */
    private String modelName(String requested, Version version, String sole) {
        if (requested != null && !requested.isBlank()) {
            return requested;
        }
        if (version == Version.V1 && sole != null) {
            return sole;
        }
        throw ApiException.badRequest("model is required");
    }

    private ApiException unknownModel(String model, String task) {
        if (registry.contains(model)) {
            return ApiException.badRequest("model '" + model + "' does not support " + task);
        }
        return new ApiException(HttpStatus.NOT_FOUND, "model '" + model + "' not found");
    }

    private static List<String> texts(EmbedRequest request, Version version) {
        if (request.images() != null && !request.images().isEmpty()) {
            throw ApiException.badRequest("image inputs are not supported");
        }
        if (request.texts() != null && request.inputs() != null) {
            throw ApiException.badRequest("set either texts or inputs, not both");
        }
        List<String> texts;
        if (request.inputs() != null) {
            if (version == Version.V1) {
                throw ApiException.badRequest("inputs is a v2 field: use texts");
            }
            texts = new ArrayList<>(request.inputs().size());
            for (int i = 0; i < request.inputs().size(); i++) {
                texts.add(inputText(request.inputs().get(i), i));
            }
        } else {
            texts = request.texts();
        }
        if (texts == null || texts.isEmpty()) {
            throw ApiException.badRequest("texts must not be empty");
        }
        for (int i = 0; i < texts.size(); i++) {
            if (texts.get(i) == null) {
                throw ApiException.badRequest("texts[" + i + "] is null");
            }
        }
        return texts;
    }

    private static String inputText(EmbedRequest.Input input, int index) {
        List<EmbedRequest.Content> content = input == null ? null : input.content();
        if (content == null || content.size() != 1) {
            throw ApiException.badRequest("inputs[" + index + "] must hold exactly one text content part");
        }
        EmbedRequest.Content part = content.getFirst();
        if (part == null || !"text".equals(part.type())) {
            throw ApiException.badRequest("inputs[" + index + "]: only text content is supported");
        }
        if (part.text() == null) {
            throw ApiException.badRequest("inputs[" + index + "] has no text");
        }
        return part.text();
    }

    private static void validateInputType(String inputType) {
        if (inputType == null || TEXT_INPUT_TYPES.contains(inputType)) {
            return;
        }
        if (inputType.equals("image")) {
            throw ApiException.badRequest("image inputs are not supported");
        }
        throw ApiException.badRequest("invalid input_type '" + inputType + "', expected one of " + TEXT_INPUT_TYPES);
    }

    /** Whether types were requested explicitly (v1 then answers by type); only float embeddings are produced. */
    private static boolean embeddingTypes(List<String> types) {
        if (types == null || types.isEmpty()) {
            return false;
        }
        for (String type : types) {
            if (!FLOAT.equals(type)) {
                throw ApiException.badRequest("embedding type '" + type + "' is not supported, only float");
            }
        }
        return true;
    }

    /** Whether over-long inputs must be refused instead of truncated. */
    private static boolean truncateNone(String truncate) {
        if (truncate == null) {
            return false;
        }
        return switch (truncate.toUpperCase(Locale.ROOT)) {
            case "END" -> false;
            case "NONE" -> true;
            case "START" -> throw ApiException.badRequest("truncate START is not supported, use END or NONE");
            default -> throw ApiException.badRequest("invalid truncate '" + truncate + "', expected NONE, START or END");
        };
    }

    private static void writeMatrix(JsonGenerator json, Embeddings embeddings) throws IOException {
        FloatWriter floats = new FloatWriter();
        float[] row = new float[embeddings.dimension()];
        json.writeStartArray();
        for (int i = 0; i < embeddings.count(); i++) {
            embeddings.copyTo(i, row, 0);
            json.writeStartArray();
            for (float value : row) {
                floats.write(json, value);
            }
            json.writeEndArray();
        }
        json.writeEndArray();
    }

    /** Cohere's rerank billing unit: one query over up to 100 documents. Clients read it, so it is always reported. */
    private static long searchUnits(int documents) {
        return (documents + 99) / 100;
    }

    private static void writeMeta(JsonGenerator json, Version version, long tokens, String billedUnit, long billed) throws IOException {
        json.writeObjectFieldStart("meta");
        json.writeObjectFieldStart("api_version");
        json.writeStringField("version", version.number);
        json.writeEndObject();
        json.writeObjectFieldStart("billed_units");
        json.writeNumberField(billedUnit, billed);
        json.writeEndObject();
        json.writeObjectFieldStart("tokens");
        json.writeNumberField("input_tokens", tokens);
        json.writeEndObject();
        json.writeEndObject();
    }
}

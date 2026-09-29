package com.ragulabs.djembed.bench;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpRequest;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * A server under test. Embeddings go through the OpenAI-compatible {@code /v1/embeddings} on both servers, so the
 * request and response bytes are the same shape; reranking uses each server's own endpoint.
 */
sealed interface Target permits Target.Djembed, Target.Tei, Target.Infinity {

    ObjectMapper JSON = new ObjectMapper();
    Duration TIMEOUT = Duration.ofMinutes(2);

    String name();

    boolean supports(Workload workload);

    HttpRequest embed(List<String> texts);

    HttpRequest rerank(String query, List<String> documents);

    /** Scores in document order. */
    float[] scores(JsonNode response, int documents);

    /** Where JVM and engine metrics can be scraped, or {@code null}. */
    URI metrics();

    /** Endpoints that answer 200 once the server is ready to serve. */
    List<URI> health();

    static float[][] embeddings(JsonNode response) {
        JsonNode data = response.path("data");
        float[][] vectors = new float[data.size()][];
        for (JsonNode item : data) {
            JsonNode values = item.path("embedding");
            float[] vector = new float[values.size()];
            for (int j = 0; j < vector.length; j++) {
                vector[j] = (float) values.get(j).asDouble();
            }
            vectors[item.path("index").asInt()] = vector;
        }
        return vectors;
    }

    private static HttpRequest post(URI uri, ObjectNode body, String apiKey) {
        try {
            HttpRequest.Builder request = HttpRequest.newBuilder(uri)
                    .timeout(TIMEOUT)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(JSON.writeValueAsBytes(body)));
            if (apiKey != null) {
                request.header("Authorization", "Bearer " + apiKey);
            }
            return request.build();
        } catch (JsonProcessingException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static ObjectNode openAiEmbeddings(String model, List<String> texts) {
        ObjectNode body = JSON.createObjectNode().put("model", model);
        ArrayNode input = body.putArray("input");
        texts.forEach(input::add);
        return body;
    }

    record Djembed(URI base, String apiKey, String embedModel, String rerankModel) implements Target {

        @Override
        public String name() {
            return "djembed";
        }

        @Override
        public boolean supports(Workload workload) {
            return true;
        }

        @Override
        public HttpRequest embed(List<String> texts) {
            return post(base.resolve("/v1/embeddings"), openAiEmbeddings(embedModel, texts), apiKey);
        }

        @Override
        public HttpRequest rerank(String query, List<String> documents) {
            ObjectNode body = JSON.createObjectNode().put("model", rerankModel).put("query", query);
            ArrayNode docs = body.putArray("documents");
            documents.forEach(docs::add);
            return post(base.resolve("/v2/rerank"), body, apiKey);
        }

        @Override
        public float[] scores(JsonNode response, int documents) {
            float[] scores = new float[documents];
            for (JsonNode result : response.path("results")) {
                scores[result.path("index").asInt()] = (float) result.path("relevance_score").asDouble();
            }
            return scores;
        }

        @Override
        public URI metrics() {
            return base.resolve("/metrics");
        }

        @Override
        public List<URI> health() {
            return List.of(base.resolve("/health"));
        }
    }

    /**
     * Infinity serves both models from one instance, under their Hugging Face ids. Its {@code /rerank} answers in the
     * Cohere shape ({@code results[].index}, {@code relevance_score}).
     */
    record Infinity(URI base, String embedModel, String rerankModel) implements Target {

        @Override
        public String name() {
            return "infinity";
        }

        @Override
        public boolean supports(Workload workload) {
            return true;
        }

        @Override
        public HttpRequest embed(List<String> texts) {
            return post(base.resolve("/embeddings"), openAiEmbeddings(embedModel, texts), null);
        }

        @Override
        public HttpRequest rerank(String query, List<String> documents) {
            ObjectNode body = JSON.createObjectNode().put("model", rerankModel).put("query", query);
            ArrayNode docs = body.putArray("documents");
            documents.forEach(docs::add);
            return post(base.resolve("/rerank"), body, null);
        }

        @Override
        public float[] scores(JsonNode response, int documents) {
            float[] scores = new float[documents];
            for (JsonNode result : response.path("results")) {
                scores[result.path("index").asInt()] = (float) result.path("relevance_score").asDouble();
            }
            return scores;
        }

        @Override
        public URI metrics() {
            return null;
        }

        @Override
        public List<URI> health() {
            return List.of(base.resolve("/health"));
        }
    }

    /** TEI serves one model per instance, hence one base URI per task; either may be absent. */
    record Tei(URI embedBase, URI rerankBase, String embedModel) implements Target {

        @Override
        public String name() {
            return "tei";
        }

        @Override
        public boolean supports(Workload workload) {
            return workload.isEmbedding() ? embedBase != null : rerankBase != null;
        }

        @Override
        public HttpRequest embed(List<String> texts) {
            return post(embedBase.resolve("/v1/embeddings"), openAiEmbeddings(embedModel, texts), null);
        }

        @Override
        public HttpRequest rerank(String query, List<String> documents) {
            ObjectNode body = JSON.createObjectNode().put("query", query);
            ArrayNode texts = body.putArray("texts");
            documents.forEach(texts::add);
            return post(rerankBase.resolve("/rerank"), body, null);
        }

        @Override
        public float[] scores(JsonNode response, int documents) {
            float[] scores = new float[documents];
            for (JsonNode result : response) {
                scores[result.path("index").asInt()] = (float) result.path("score").asDouble();
            }
            return scores;
        }

        @Override
        public URI metrics() {
            return null;
        }

        @Override
        public List<URI> health() {
            return Stream.of(embedBase, rerankBase).filter(Objects::nonNull).map(base -> base.resolve("/health")).toList();
        }
    }
}

package com.ragulabs.djembed.server.cohere;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.linecorp.armeria.client.WebClient;
import com.linecorp.armeria.common.AggregatedHttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.common.MediaType;
import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.testing.junit5.server.ServerExtension;
import com.ragulabs.djembed.server.FakeEngines;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CohereServiceTest {

    @RegisterExtension
    static final ServerExtension server = new ServerExtension() {
        @Override
        protected void configure(ServerBuilder sb) {
            sb.annotatedService(new CohereService(FakeEngines.registry()));
        }
    };

    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void v1EmbedReturnsFloatMatrix() throws IOException {
        JsonNode body = ok("/v1/embed", """
                {"model": "fake-embed", "texts": ["ab", "abcd"], "input_type": "search_document"}""");

        assertEquals("embeddings_floats", body.path("response_type").asText());
        assertEquals(2, body.path("embeddings").size());
        assertEquals(2.0, body.path("embeddings").get(0).get(0).asDouble());
        assertEquals(4.0, body.path("embeddings").get(1).get(0).asDouble());
        assertEquals(0.5, body.path("embeddings").get(1).get(2).asDouble());
        assertEquals("abcd", body.path("texts").get(1).asText());
        assertEquals("1", body.path("meta").path("api_version").path("version").asText());
        assertEquals(6, body.path("meta").path("billed_units").path("input_tokens").asInt());
        assertFalse(body.path("id").asText().isEmpty());
    }

    @Test
    void v1EmbedWithTypesAnswersByType() throws IOException {
        JsonNode body = ok("/v1/embed", """
                {"texts": ["ab"], "embedding_types": ["float"]}""");

        assertEquals("embeddings_by_type", body.path("response_type").asText());
        assertEquals(2.0, body.path("embeddings").path("float").get(0).get(0).asDouble());
    }

    @Test
    void v2EmbedAcceptsTextsAndInputs() throws IOException {
        JsonNode fromTexts = ok("/v2/embed", """
                {"model": "fake-embed", "input_type": "search_query", "texts": ["abc"]}""");
        JsonNode fromInputs = ok("/v2/embed", """
                {"model": "fake-embed", "input_type": "search_query",
                 "inputs": [{"content": [{"type": "text", "text": "abc"}]}]}""");

        for (JsonNode body : new JsonNode[]{fromTexts, fromInputs}) {
            assertEquals(3.0, body.path("embeddings").path("float").get(0).get(0).asDouble());
            assertEquals("2", body.path("meta").path("api_version").path("version").asText());
            assertTrue(body.path("response_type").isMissingNode());
        }
    }

    @Test
    void v2RequiresTheModel() throws IOException {
        error("/v2/embed", """
                {"texts": ["a"]}""", HttpStatus.BAD_REQUEST, "model is required");
    }

    @Test
    void unknownAndMismatchedModels() throws IOException {
        error("/v2/embed", """
                {"model": "nope", "texts": ["a"]}""", HttpStatus.NOT_FOUND, "model 'nope' not found");
        error("/v2/embed", """
                {"model": "fake-rerank", "texts": ["a"]}""", HttpStatus.BAD_REQUEST, "does not support embed");
    }

    @Test
    void truncateNoneRefusesLongInputs() throws IOException {
        String longText = "x".repeat(FakeEngines.LIMIT + 1);
        ok("/v2/embed", """
                {"model": "fake-embed", "texts": ["%s"], "truncate": "END"}""".formatted(longText));
        error("/v2/embed", """
                {"model": "fake-embed", "texts": ["ok", "%s"], "truncate": "NONE"}""".formatted(longText),
                HttpStatus.BAD_REQUEST, "texts[1] exceeds the model limit");
    }

    @Test
    void refusesWhatCannotBeHonoured() throws IOException {
        error("/v2/embed", """
                {"model": "fake-embed", "texts": ["a"], "embedding_types": ["int8"]}""", HttpStatus.BAD_REQUEST, "'int8'");
        error("/v2/embed", """
                {"model": "fake-embed", "texts": ["a"], "truncate": "START"}""", HttpStatus.BAD_REQUEST, "START");
        error("/v2/embed", """
                {"model": "fake-embed", "images": ["data:image/png;base64,AA=="]}""", HttpStatus.BAD_REQUEST, "image");
        error("/v2/embed", """
                {"model": "fake-embed", "texts": ["a"], "output_dimension": 256}""", HttpStatus.BAD_REQUEST, "3 dimensions");
        error("/v2/embed", """
                {"model": "fake-embed", "texts": ["a"], "max_tokens": 5}""", HttpStatus.BAD_REQUEST, "max_tokens");
        error("/v2/embed", """
                {"model": "fake-embed", "texts": []}""", HttpStatus.BAD_REQUEST, "must not be empty");
    }

    @Test
    void overloadIsServiceUnavailable() throws IOException {
        // Thrown on submission (embed) and delivered through the future (rerank): both surface as 503.
        error("/v2/embed", """
                {"model": "fake-embed", "texts": ["overload"]}""", HttpStatus.SERVICE_UNAVAILABLE, "queue full");
        error("/v2/rerank", """
                {"model": "fake-rerank", "query": "overload", "documents": ["a"]}""", HttpStatus.SERVICE_UNAVAILABLE, "queue full");
    }

    @Test
    void malformedJsonIsABadRequest() throws IOException {
        error("/v1/embed", "{\"texts\": [", HttpStatus.BAD_REQUEST, "invalid request body");
    }

    @Test
    void rerankSortsByRelevanceAndHonoursTopN() throws IOException {
        JsonNode body = ok("/v2/rerank", """
                {"model": "fake-rerank", "query": "cat", "top_n": 2,
                 "documents": ["dogs", "a cat sleeps", "a very long document about nothing"]}""");

        JsonNode results = body.path("results");
        assertEquals(2, results.size());
        assertEquals(1, results.get(0).path("index").asInt());
        assertEquals(0.9, results.get(0).path("relevance_score").asDouble(), 1e-6);
        assertEquals(2, results.get(1).path("index").asInt());
        assertTrue(results.get(0).path("document").isMissingNode());
        assertEquals("2", body.path("meta").path("api_version").path("version").asText());
        assertEquals(1, body.path("meta").path("billed_units").path("search_units").asInt());
    }

    @Test
    void v1RerankReturnsDocumentsAndAcceptsObjects() throws IOException {
        JsonNode body = ok("/v1/rerank", """
                {"query": "cat", "return_documents": true,
                 "documents": [{"text": "dogs"}, "a cat"]}""");

        JsonNode first = body.path("results").get(0);
        assertEquals(1, first.path("index").asInt());
        assertEquals("a cat", first.path("document").path("text").asText());
        assertEquals("dogs", body.path("results").get(1).path("document").path("text").asText());
    }

    @Test
    void rerankValidation() throws IOException {
        error("/v2/rerank", """
                {"model": "fake-rerank", "query": "q", "documents": []}""", HttpStatus.BAD_REQUEST, "documents must not be empty");
        error("/v2/rerank", """
                {"model": "fake-rerank", "documents": ["a"]}""", HttpStatus.BAD_REQUEST, "query is required");
        error("/v1/rerank", """
                {"query": "q", "documents": [{"title": "t"}]}""", HttpStatus.BAD_REQUEST, "documents[0] has no text");
        error("/v1/rerank", """
                {"query": "q", "documents": ["a"], "rank_fields": ["title"]}""", HttpStatus.BAD_REQUEST, "rank_fields");
        error("/v2/rerank", """
                {"model": "fake-rerank", "query": "q", "documents": ["a"], "top_n": 0}""", HttpStatus.BAD_REQUEST, "top_n");
    }

    private static JsonNode ok(String path, String body) throws IOException {
        AggregatedHttpResponse response = post(path, body);
        assertEquals(HttpStatus.OK, response.status(), response.contentUtf8());
        assertEquals(MediaType.JSON_UTF_8, response.contentType());
        return JSON.readTree(response.contentUtf8());
    }

    private static void error(String path, String body, HttpStatus status, String messagePart) throws IOException {
        AggregatedHttpResponse response = post(path, body);
        assertEquals(status, response.status(), response.contentUtf8());
        JsonNode json = JSON.readTree(response.contentUtf8());
        assertTrue(json.path("message").asText().contains(messagePart), json.toString());
        assertFalse(json.path("id").asText().isEmpty());
    }

    private static AggregatedHttpResponse post(String path, String body) {
        return WebClient.of(server.httpUri()).prepare().post(path).content(MediaType.JSON, body).execute().aggregate().join();
    }
}

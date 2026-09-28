package dev.fgnm.djembed.server.openai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.linecorp.armeria.client.WebClient;
import com.linecorp.armeria.common.AggregatedHttpResponse;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.common.MediaType;
import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.testing.junit5.server.ServerExtension;
import dev.fgnm.djembed.server.FakeEngines;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OpenAiServiceTest {

    @RegisterExtension
    static final ServerExtension server = new ServerExtension() {
        @Override
        protected void configure(ServerBuilder sb) {
            sb.annotatedService(new OpenAiService(FakeEngines.registry()));
        }
    };

    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void floatEmbeddings() throws IOException {
        JsonNode body = ok("""
                {"model": "fake-embed", "input": ["ab", "abcd"]}""");

        assertEquals("list", body.path("object").asText());
        assertEquals("fake-embed", body.path("model").asText());
        JsonNode second = body.path("data").get(1);
        assertEquals("embedding", second.path("object").asText());
        assertEquals(1, second.path("index").asInt());
        assertEquals(4.0, second.path("embedding").get(0).asDouble());
        assertEquals(0.5, second.path("embedding").get(2).asDouble());
        assertEquals(6, body.path("usage").path("prompt_tokens").asInt());
        assertEquals(6, body.path("usage").path("total_tokens").asInt());
    }

    @Test
    void singleStringInput() throws IOException {
        JsonNode body = ok("""
                {"model": "fake-embed", "input": "abc"}""");

        assertEquals(1, body.path("data").size());
        assertEquals(3.0, body.path("data").get(0).path("embedding").get(0).asDouble());
    }

    @Test
    void base64IsLittleEndianFloat32() throws IOException {
        JsonNode body = ok("""
                {"model": "fake-embed", "input": ["abcd"], "encoding_format": "base64"}""");

        byte[] bytes = Base64.getDecoder().decode(body.path("data").get(0).path("embedding").asText());
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(12, bytes.length);
        assertEquals(4f, buffer.getFloat(0));
        assertEquals(1f, buffer.getFloat(4));
        assertEquals(0.5f, buffer.getFloat(8));
    }

    @Test
    void listsModels() throws IOException {
        AggregatedHttpResponse response = WebClient.of(server.httpUri()).get("/v1/models").aggregate().join();
        JsonNode data = JSON.readTree(response.contentUtf8()).path("data");

        assertEquals(2, data.size());
        assertEquals("fake-embed", data.get(0).path("id").asText());
        assertEquals("fake-rerank", data.get(1).path("id").asText());
        assertEquals("djembed", data.get(0).path("owned_by").asText());
    }

    @Test
    void errorsUseTheOpenAiShape() throws IOException {
        JsonNode missing = error("""
                {"model": "nope", "input": "a"}""", HttpStatus.NOT_FOUND);
        assertEquals("model_not_found", missing.path("code").asText());
        assertEquals("invalid_request_error", missing.path("type").asText());

        assertTrue(error("""
                {"model": "fake-embed", "input": [[1, 2, 3]]}""", HttpStatus.BAD_REQUEST)
                .path("message").asText().contains("token-id inputs are not supported"));
        assertTrue(error("""
                {"model": "fake-embed", "input": ["%s"]}""".formatted("x".repeat(FakeEngines.LIMIT + 1)), HttpStatus.BAD_REQUEST)
                .path("message").asText().contains("maximum context length"));
        assertTrue(error("""
                {"model": "fake-embed", "input": "a", "dimensions": 2}""", HttpStatus.BAD_REQUEST)
                .path("message").asText().contains("3 dimensions"));
        assertTrue(error("""
                {"model": "fake-embed", "input": [""]}""", HttpStatus.BAD_REQUEST)
                .path("message").asText().contains("non-empty"));
        assertTrue(error("""
                {"model": "fake-rerank", "input": "a"}""", HttpStatus.BAD_REQUEST)
                .path("message").asText().contains("does not support embeddings"));

        JsonNode overloaded = error("""
                {"model": "fake-embed", "input": "overload"}""", HttpStatus.SERVICE_UNAVAILABLE);
        assertEquals("server_error", overloaded.path("type").asText());
    }

    private static JsonNode ok(String body) throws IOException {
        AggregatedHttpResponse response = post(body);
        assertEquals(HttpStatus.OK, response.status(), response.contentUtf8());
        return JSON.readTree(response.contentUtf8());
    }

    private static JsonNode error(String body, HttpStatus status) throws IOException {
        AggregatedHttpResponse response = post(body);
        assertEquals(status, response.status(), response.contentUtf8());
        JsonNode error = JSON.readTree(response.contentUtf8()).path("error");
        assertTrue(error.isObject(), response.contentUtf8());
        return error;
    }

    private static AggregatedHttpResponse post(String body) {
        return WebClient.of(server.httpUri()).prepare().post("/v1/embeddings").content(MediaType.JSON, body)
                .execute().aggregate().join();
    }
}

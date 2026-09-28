package com.ragulabs.djembed.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.linecorp.armeria.client.WebClient;
import com.linecorp.armeria.common.AggregatedHttpResponse;
import com.linecorp.armeria.common.HttpHeaderNames;
import com.linecorp.armeria.common.HttpStatus;
import com.linecorp.armeria.common.MediaType;
import com.linecorp.armeria.server.Server;
import com.ragulabs.djembed.core.EngineObserver;
import com.ragulabs.djembed.server.config.DjembedConfig;
import com.ragulabs.djembed.server.config.ServerConfig;
import com.ragulabs.djembed.server.metrics.DjembedMetrics;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The assembled server: routes, API-key protection and metrics.
 */
class DjembedServerTest {

    private static final String KEY = "s3cret";
    private static final ObjectMapper JSON = new ObjectMapper();

    private static Server server;
    private static DjembedMetrics metrics;
    private static WebClient client;

    @BeforeAll
    static void start() {
        metrics = new DjembedMetrics();
        DjembedConfig config = new DjembedConfig(new ServerConfig("127.0.0.1", 0, null, null, null), List.of());
        server = Djembed.server(config, FakeEngines.registry(), metrics, List.of(KEY, "other"));
        server.start().join();
        client = WebClient.of("http://127.0.0.1:" + server.activeLocalPort());
    }

    @AfterAll
    static void stop() {
        server.stop().join();
    }

    @Test
    void apiRoutesRequireAKey() throws IOException {
        AggregatedHttpResponse cohere = post("/v2/embed", """
                {"model": "fake-embed", "texts": ["a"]}""", null);
        assertEquals(HttpStatus.UNAUTHORIZED, cohere.status());
        assertTrue(JSON.readTree(cohere.contentUtf8()).path("message").asText().contains("API key"));

        AggregatedHttpResponse openAi = post("/v1/embeddings", """
                {"model": "fake-embed", "input": "a"}""", "Bearer wrong");
        assertEquals(HttpStatus.UNAUTHORIZED, openAi.status());
        assertEquals("invalid_api_key", JSON.readTree(openAi.contentUtf8()).path("error").path("code").asText());

        assertEquals(HttpStatus.OK, post("/v2/embed", """
                {"model": "fake-embed", "texts": ["a"]}""", "Bearer " + KEY).status());
        assertEquals(HttpStatus.OK, post("/v1/embeddings", """
                {"model": "fake-embed", "input": "a"}""", "bearer other").status());
    }

    @Test
    void healthAndMetricsAreOpen() {
        post("/v2/rerank", """
                {"model": "fake-rerank", "query": "q", "documents": ["a"]}""", "Bearer " + KEY);

        assertEquals(HttpStatus.OK, client.get("/health").aggregate().join().status());
        AggregatedHttpResponse scrape = client.get("/metrics").aggregate().join();
        assertEquals(HttpStatus.OK, scrape.status());
        String text = scrape.contentUtf8();
        assertTrue(text.contains("djembed_queue_inputs{model=\"fake-embed\"}"), text);
        assertTrue(text.contains("djembed_http_requests_total"), text);
        assertTrue(text.contains("jvm_gc_"), text);
        assertTrue(text.contains("djembed_jvm_allocated_bytes_total"), text);
    }

    @Test
    void forwardPassesAreRecorded() {
        EngineObserver observer = metrics.observer("probe");
        observer.forwardPass(4, 10, 25, 1_000_000);

        String text = client.get("/metrics").aggregate().join().contentUtf8();
        assertTrue(text.contains("djembed_tokens_total{kind=\"real\",model=\"probe\"} 25.0"), text);
        assertTrue(text.contains("djembed_tokens_total{kind=\"padding\",model=\"probe\"} 15.0"), text);
        assertTrue(text.contains("djembed_forward_pass_seconds_count{model=\"probe\"} 1"), text);
    }

    private static AggregatedHttpResponse post(String path, String body, String authorization) {
        var request = client.prepare().post(path).content(MediaType.JSON, body);
        if (authorization != null) {
            request.header(HttpHeaderNames.AUTHORIZATION, authorization);
        }
        return request.execute().aggregate().join();
    }
}

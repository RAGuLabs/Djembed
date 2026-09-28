package com.ragulabs.djembed.bench;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LoadRunnerTest {

    private HttpServer server;
    private final AtomicInteger calls = new AtomicInteger();

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.createContext("/ok", exchange -> {
            calls.incrementAndGet();
            try {
                Thread.sleep(5);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            byte[] body = "{}".getBytes();
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.createContext("/fail", exchange -> {
            exchange.sendResponseHeaders(503, -1);
            exchange.close();
        });
        server.start();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    @Test
    void measuresOnlyTheWindowAndOpensItOnce() throws Exception {
        AtomicInteger opened = new AtomicInteger();
        AtomicInteger closed = new AtomicInteger();

        LoadRunner.Measurement m = LoadRunner.run(client(), List.of(request("/ok")), 4,
                Duration.ofMillis(200), Duration.ofMillis(500), new LoadRunner.Window() {
                    @Override
                    public void opened() {
                        opened.incrementAndGet();
                    }

                    @Override
                    public void closed() {
                        closed.incrementAndGet();
                    }
                });

        assertEquals(1, opened.get());
        assertEquals(1, closed.get());
        assertEquals(0, m.errors());
        assertTrue(m.requests() > 50, "requests: " + m.requests());
        assertTrue(m.requests() < calls.get(), "warm-up requests must not count");
        assertTrue(m.latency().getValueAtPercentile(50) >= 5_000, "latency in µs, at least the server's 5 ms");
    }

    @Test
    void countsFailures() throws Exception {
        LoadRunner.Measurement m = LoadRunner.run(client(), List.of(request("/fail")), 2,
                Duration.ZERO, Duration.ofMillis(200), new LoadRunner.Window() {
                    @Override
                    public void opened() {
                    }

                    @Override
                    public void closed() {
                    }
                });

        assertEquals(0, m.requests());
        assertTrue(m.errors() > 0);
        assertEquals("HTTP 503", m.firstError());
    }

    private HttpRequest request(String path) {
        return HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.getAddress().getPort() + path))
                .POST(HttpRequest.BodyPublishers.ofString("{}")).build();
    }

    private static HttpClient client() {
        return HttpClient.newBuilder().executor(Executors.newVirtualThreadPerTaskExecutor()).build();
    }
}

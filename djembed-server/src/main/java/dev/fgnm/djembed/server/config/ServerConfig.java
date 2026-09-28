package dev.fgnm.djembed.server.config;

import java.util.List;

/**
 * Settings of the HTTP frontend.
 *
 * @param apiKeys bearer tokens accepted on the API routes; none means the API is open
 */
public record ServerConfig(String host, Integer port, Long maxRequestBytes, Long requestTimeoutMs, List<String> apiKeys) {

    static final ServerConfig DEFAULT = new ServerConfig(null, null, null, null, null);

    private static final String DEFAULT_HOST = "0.0.0.0";
    private static final int DEFAULT_PORT = 8080;
    private static final long DEFAULT_MAX_REQUEST_BYTES = 32L * 1024 * 1024;
    // Large embedding batches queue behind other work under load; Armeria's 10 s default is too tight for them.
    private static final long DEFAULT_REQUEST_TIMEOUT_MS = 60_000;

    public ServerConfig {
        host = host != null && !host.isBlank() ? host : DEFAULT_HOST;
        port = port != null ? port : DEFAULT_PORT;
        maxRequestBytes = maxRequestBytes != null ? maxRequestBytes : DEFAULT_MAX_REQUEST_BYTES;
        requestTimeoutMs = requestTimeoutMs != null ? requestTimeoutMs : DEFAULT_REQUEST_TIMEOUT_MS;
        apiKeys = apiKeys != null ? List.copyOf(apiKeys) : List.of();
        if (apiKeys.stream().anyMatch(String::isBlank)) {
            throw new IllegalArgumentException("server.api-keys must not contain blank keys");
        }
        if (port < 0 || port > 65535) {
            throw new IllegalArgumentException("server.port out of range: " + port);
        }
        if (maxRequestBytes < 1) {
            throw new IllegalArgumentException("server.max-request-bytes must be positive: " + maxRequestBytes);
        }
        if (requestTimeoutMs < 0) {
            throw new IllegalArgumentException("server.request-timeout-ms must be >= 0 (0 disables it): " + requestTimeoutMs);
        }
    }
}

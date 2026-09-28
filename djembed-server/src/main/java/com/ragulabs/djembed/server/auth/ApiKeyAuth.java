package com.ragulabs.djembed.server.auth;

import com.linecorp.armeria.common.HttpHeaderNames;
import com.linecorp.armeria.common.HttpRequest;
import com.linecorp.armeria.common.HttpResponse;
import com.linecorp.armeria.server.HttpService;
import com.linecorp.armeria.server.ServiceRequestContext;
import com.linecorp.armeria.server.SimpleDecoratingHttpService;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collection;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Requires {@code Authorization: Bearer <key>} with one of the configured keys, as both Cohere and OpenAI clients
 * send it.
 *
 * <p>Only SHA-256 digests of the keys are kept, and a presented key is compared against every one of them in
 * constant time, so neither the comparison nor the number of keys tried leaks through timing.
 */
public final class ApiKeyAuth {

    private static final String BEARER = "Bearer ";

    private final List<byte[]> digests;

    private ApiKeyAuth(List<byte[]> digests) {
        this.digests = digests;
    }

    public static ApiKeyAuth of(Collection<String> keys) {
        if (keys.isEmpty()) {
            throw new IllegalArgumentException("at least one API key is required");
        }
        return new ApiKeyAuth(keys.stream().map(ApiKeyAuth::sha256).toList());
    }

    boolean accepts(String authorization) {
        if (authorization == null || !authorization.regionMatches(true, 0, BEARER, 0, BEARER.length())) {
            return false;
        }
        byte[] presented = sha256(authorization.substring(BEARER.length()).trim());
        boolean match = false;
        for (byte[] digest : digests) {
            match |= MessageDigest.isEqual(digest, presented);
        }
        return match;
    }

    /**
     * Decorator answering unauthorised requests with {@code unauthorized}, so each API keeps its own error format.
     */
    public Function<? super HttpService, ? extends HttpService> decorator(
            BiFunction<ServiceRequestContext, String, HttpResponse> unauthorized) {
        return delegate -> new SimpleDecoratingHttpService(delegate) {
            @Override
            public HttpResponse serve(ServiceRequestContext ctx, HttpRequest req) throws Exception {
                if (accepts(req.headers().get(HttpHeaderNames.AUTHORIZATION))) {
                    return unwrap().serve(ctx, req);
                }
                return unauthorized.apply(ctx, "invalid or missing API key");
            }
        };
    }

    private static byte[] sha256(String key) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required of every JVM", e);
        }
    }
}

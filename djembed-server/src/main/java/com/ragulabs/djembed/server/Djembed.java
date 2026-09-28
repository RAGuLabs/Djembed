package com.ragulabs.djembed.server;

import com.linecorp.armeria.common.metric.MeterIdPrefixFunction;
import com.linecorp.armeria.server.HttpService;
import com.linecorp.armeria.server.Server;
import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.server.healthcheck.HealthCheckService;
import com.linecorp.armeria.server.metric.MetricCollectingService;
import com.ragulabs.djembed.server.auth.ApiKeyAuth;
import com.ragulabs.djembed.server.cohere.CohereExceptionHandler;
import com.ragulabs.djembed.server.cohere.CohereService;
import com.ragulabs.djembed.server.config.ConfigLoader;
import com.ragulabs.djembed.server.config.DjembedConfig;
import com.ragulabs.djembed.server.metrics.DjembedMetrics;
import com.ragulabs.djembed.server.openai.OpenAiExceptionHandler;
import com.ragulabs.djembed.server.openai.OpenAiService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public final class Djembed {

    private static final Logger log = LoggerFactory.getLogger(Djembed.class);

    private static final String CONFIG_ENV = "DJEMBED_CONFIG";
    private static final String API_KEYS_ENV = "DJEMBED_API_KEYS";
    private static final String DEFAULT_CONFIG = "djembed.yaml";

    private Djembed() {
    }

    public static void main(String[] args) {
        Path configPath = Path.of(resolveConfigLocation(args));
        DjembedConfig config = ConfigLoader.load(configPath);
        log.info("Loading {} model(s) from {}", config.models().size(), configPath.toAbsolutePath());

        DjembedMetrics metrics = new DjembedMetrics();
        ModelRegistry registry = ModelRegistry.load(config.models(), metrics::observer);
        Server server = server(config, registry, metrics, apiKeys(config));

        // Stop accepting and drain in-flight requests before the engines they use are released.
        Runtime.getRuntime().addShutdownHook(Thread.ofPlatform().name("djembed-shutdown").unstarted(() -> {
            server.stop().join();
            registry.close();
        }));
        server.start().join();
        log.info("Djembed listening on {}", server.activeLocalPort());
    }

    static Server server(DjembedConfig config, ModelRegistry registry, DjembedMetrics metrics, List<String> apiKeys) {
        registry.embedders().forEach((name, engine) -> metrics.queue(name, engine::queuedInputs));
        registry.rerankers().forEach((name, engine) -> metrics.queue(name, engine::queuedInputs));

        ServerBuilder sb = Server.builder()
                .http(new InetSocketAddress(config.server().host(), config.server().port()))
                .maxRequestLength(config.server().maxRequestBytes())
                .requestTimeoutMillis(config.server().requestTimeoutMs())
                .meterRegistry(metrics.registry())
                .service("/health", HealthCheckService.of())
                .service("/metrics", metrics.exposition());

        Function<? super HttpService, ? extends HttpService> cohereAuth = Function.identity();
        Function<? super HttpService, ? extends HttpService> openAiAuth = Function.identity();
        if (apiKeys.isEmpty()) {
            log.warn("No API keys configured: the API is open to anyone who can reach port {}", config.server().port());
        } else {
            ApiKeyAuth auth = ApiKeyAuth.of(apiKeys);
            cohereAuth = auth.decorator(CohereExceptionHandler::unauthorized);
            openAiAuth = auth.decorator(OpenAiExceptionHandler::unauthorized);
        }
        Function<? super HttpService, ? extends HttpService> httpMetrics =
                MetricCollectingService.newDecorator(MeterIdPrefixFunction.ofDefault("djembed.http"));

        sb.annotatedService().decorator(cohereAuth).decorator(httpMetrics).build(new CohereService(registry));
        sb.annotatedService().decorator(openAiAuth).decorator(httpMetrics).build(new OpenAiService(registry));
        return sb.build();
    }

    /** Keys from the configuration plus the comma-separated {@value #API_KEYS_ENV}, which keeps them out of files. */
    private static List<String> apiKeys(DjembedConfig config) {
        List<String> keys = new ArrayList<>(config.server().apiKeys());
        String fromEnv = System.getenv(API_KEYS_ENV);
        if (fromEnv != null) {
            Arrays.stream(fromEnv.split(",")).map(String::trim).filter(key -> !key.isEmpty()).forEach(keys::add);
        }
        return keys;
    }

    private static String resolveConfigLocation(String[] args) {
        if (args.length > 0) {
            return args[0];
        }
        String fromEnv = System.getenv(CONFIG_ENV);
        return fromEnv != null && !fromEnv.isBlank() ? fromEnv : DEFAULT_CONFIG;
    }
}

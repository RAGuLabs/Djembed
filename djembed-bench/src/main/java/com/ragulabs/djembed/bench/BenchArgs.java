package com.ragulabs.djembed.bench;

import java.net.URI;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Command line of the benchmark; see {@link #USAGE}.
 */
record BenchArgs(
        URI djembed,
        URI teiEmbed,
        URI teiRerank,
        String apiKey,
        String embedModel,
        String rerankModel,
        List<Workload> workloads,
        List<Integer> concurrency,
        Duration warmup,
        Duration duration,
        long seed,
        Path out,
        Integer gpu) {

    static final String USAGE = """
            Usage: bench [options]
              --djembed <url>        Djembed base URL (embed and rerank)
              --tei-embed <url>      TEI instance serving the embedding model
              --tei-rerank <url>     TEI instance serving the rerank model
              --api-key <key>        Djembed API key, if it has any
              --embed-model <name>   Djembed embedding model name (default bge-m3)
              --rerank-model <name>  Djembed rerank model name (default bge-reranker-v2-m3)
              --workloads <list>     query,ingest,rerank (default all)
              --concurrency <list>   closed-loop client counts (default 1,8,32,128)
              --warmup <duration>    per run, e.g. 15s (default 15s)
              --duration <duration>  measured window per run, e.g. 60s (default 60s)
              --seed <n>             corpus seed (default 42)
              --out <dir>            results directory (default bench-results)
              --gpu <index>          sample this GPU with nvidia-smi (only when run on the GPU host)
            """;

    private static final Set<String> FLAGS = Set.of("--djembed", "--tei-embed", "--tei-rerank", "--api-key", "--embed-model",
            "--rerank-model", "--workloads", "--concurrency", "--warmup", "--duration", "--seed", "--out", "--gpu");

    static BenchArgs parse(String[] args) {
        Map<String, String> values = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String flag = args[i];
            if (!FLAGS.contains(flag)) {
                throw new IllegalArgumentException("Unknown option " + flag);
            }
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException(flag + " needs a value");
            }
            values.put(flag, args[++i]);
        }
        BenchArgs parsed = new BenchArgs(
                uri(values.get("--djembed")),
                uri(values.get("--tei-embed")),
                uri(values.get("--tei-rerank")),
                values.get("--api-key"),
                values.getOrDefault("--embed-model", "bge-m3"),
                values.getOrDefault("--rerank-model", "bge-reranker-v2-m3"),
                Arrays.stream(values.getOrDefault("--workloads", "query,ingest,rerank").split(",")).map(Workload::parse).toList(),
                Arrays.stream(values.getOrDefault("--concurrency", "1,8,32,128").split(",")).map(String::trim).map(Integer::parseInt).toList(),
                duration(values.getOrDefault("--warmup", "15s")),
                duration(values.getOrDefault("--duration", "60s")),
                Long.parseLong(values.getOrDefault("--seed", "42")),
                Path.of(values.getOrDefault("--out", "bench-results")),
                values.containsKey("--gpu") ? Integer.valueOf(values.get("--gpu")) : null);
        if (parsed.djembed == null && parsed.teiEmbed == null && parsed.teiRerank == null) {
            throw new IllegalArgumentException("Give at least one of --djembed, --tei-embed, --tei-rerank");
        }
        if (parsed.concurrency.stream().anyMatch(c -> c < 1)) {
            throw new IllegalArgumentException("Concurrency levels must be positive");
        }
        return parsed;
    }

    List<Target> targets() {
        List<Target> targets = new ArrayList<>(2);
        if (djembed != null) {
            targets.add(new Target.Djembed(djembed, apiKey, embedModel, rerankModel));
        }
        if (teiEmbed != null || teiRerank != null) {
            targets.add(new Target.Tei(teiEmbed, teiRerank, embedModel));
        }
        return targets;
    }

    private static URI uri(String value) {
        return value == null ? null : URI.create(value.endsWith("/") ? value.substring(0, value.length() - 1) : value);
    }

    /** {@code 500ms}, {@code 15s}, {@code 2m}. */
    static Duration duration(String value) {
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.endsWith("ms")) {
            return Duration.ofMillis(Long.parseLong(v.substring(0, v.length() - 2)));
        }
        if (v.endsWith("s")) {
            return Duration.ofSeconds(Long.parseLong(v.substring(0, v.length() - 1)));
        }
        if (v.endsWith("m")) {
            return Duration.ofMinutes(Long.parseLong(v.substring(0, v.length() - 1)));
        }
        throw new IllegalArgumentException("Invalid duration '" + value + "', use e.g. 500ms, 15s, 2m");
    }
}

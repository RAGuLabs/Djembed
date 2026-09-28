package com.ragulabs.djembed.bench;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Prints results as they come and writes {@code results.md} and {@code results.json}.
 */
final class Report {

    private static final String ROW = "%-8s %5s %9s %10s %11s %8s %8s %8s %7s %5s %11s %8s%n";

    private Report() {
    }

    static void header(Workload workload, double tokensPerInput) {
        System.out.printf(Locale.ROOT, "%n== %s: %d input(s) per request%s%n", workload.label(), workload.inputs,
                Double.isNaN(tokensPerInput) ? "" : String.format(Locale.ROOT, ", ~%.0f tokens per input", tokensPerInput));
        System.out.printf(Locale.ROOT, ROW, "target", "conc", "req/s", "inputs/s", "tokens/s", "p50 ms", "p90 ms", "p99 ms",
                "errors", "GPU%", "alloc/req", "GC ms");
    }

    static void row(Result r) {
        System.out.printf(Locale.ROOT, ROW, r.target(), r.concurrency(), f1(r.requestsPerSecond()), f0(r.inputsPerSecond()),
                f0(r.tokensPerSecond()), f1(r.p50()), f1(r.p90()), f1(r.p99()), r.errors(), f0(r.gpuUtilization()),
                bytes(r.allocatedPerRequest()), f1(r.gcPauseMs()));
        if (r.firstError() != null) {
            System.out.println("         first error: " + r.firstError());
        }
    }

    static void write(Path dir, BenchArgs args, String gpu, Correctness.Report check, List<Result> results) throws IOException {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("results.md"), markdown(args, gpu, check, results));
        Files.writeString(dir.resolve("results.json"), json(args, gpu, check, results));
        System.out.println("\nResults written to " + dir.toAbsolutePath());
    }

    static String markdown(BenchArgs args, String gpu, Correctness.Report check, List<Result> results) {
        StringBuilder md = new StringBuilder();
        md.append("# Djembed vs TEI\n\n");
        md.append("- GPU: ").append(gpu == null ? "not sampled" : gpu).append('\n');
        md.append("- Java ").append(Runtime.version()).append(", seed ").append(args.seed())
                .append(", warm-up ").append(args.warmup().toSeconds()).append(" s, measured ")
                .append(args.duration().toSeconds()).append(" s per run\n");
        check.minCosine().forEach((target, cos) -> md.append(String.format(Locale.ROOT,
                "- Embeddings, min cosine vs reference: %s %.6f%n", target, cos)));
        check.maxScoreDiff().forEach((target, diff) -> md.append(String.format(Locale.ROOT,
                "- Rerank, max score difference vs reference: %s %.2e, planted document ranked first: %s%n",
                target, diff, check.sameTop().get(target))));

        for (Workload workload : args.workloads()) {
            List<Result> rows = results.stream().filter(r -> r.workload() == workload).toList();
            if (rows.isEmpty()) {
                continue;
            }
            md.append("\n## ").append(workload.label()).append(" (").append(workload.inputs).append(" input(s) per request)\n\n");
            md.append("| target | concurrency | req/s | inputs/s | tokens/s | p50 ms | p90 ms | p99 ms | errors | GPU % |\n");
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            for (Result r : rows) {
                md.append(String.format(Locale.ROOT, "| %s | %d | %s | %s | %s | %s | %s | %s | %d | %s |%n", r.target(), r.concurrency(),
                        f1(r.requestsPerSecond()), f0(r.inputsPerSecond()), f0(r.tokensPerSecond()),
                        f1(r.p50()), f1(r.p90()), f1(r.p99()), r.errors(), f0(r.gpuUtilization())));
            }
        }

        String comparison = comparison(args, results);
        if (!comparison.isEmpty()) {
            md.append("\n## Djembed relative to TEI\n\n");
            md.append("Throughput above 1.00× and p99 below 1.00× favour Djembed. Runs where either server returned errors are not compared.\n\n");
            md.append("| workload | concurrency | throughput | p99 latency |\n|---|---:|---:|---:|\n");
            md.append(comparison);
        }
        return md.toString();
    }

    private static String comparison(BenchArgs args, List<Result> results) {
        StringBuilder rows = new StringBuilder();
        for (Workload workload : args.workloads()) {
            for (int concurrency : args.concurrency()) {
                Optional<Result> djembed = find(results, workload, concurrency, "djembed");
                Optional<Result> tei = find(results, workload, concurrency, "tei");
                if (djembed.isEmpty() || tei.isEmpty()) {
                    continue;
                }
                if (!valid(djembed.get()) || !valid(tei.get())) {
                    rows.append(String.format(Locale.ROOT, "| %s | %d | invalid: errors | invalid: errors |%n", workload.label(), concurrency));
                    continue;
                }
                rows.append(String.format(Locale.ROOT, "| %s | %d | %.2f× | %.2f× |%n", workload.label(), concurrency,
                        djembed.get().requestsPerSecond() / tei.get().requestsPerSecond(),
                        djembed.get().p99() / tei.get().p99()));
            }
        }
        return rows.toString();
    }

    static void summary(BenchArgs args, List<Result> results) {
        String comparison = comparison(args, results);
        if (!comparison.isEmpty()) {
            System.out.println("\n== Djembed relative to TEI (throughput > 1 and p99 < 1 favour Djembed)");
            System.out.print(comparison.replace("|", " ").replaceAll(" +", " ").replace("\n ", "\n"));
        }
    }

    /** A run is comparable only if every request succeeded: a server refusing work is not serving it faster. */
    static boolean valid(Result result) {
        return result.errors() == 0 && result.requests() > 0;
    }

    private static Optional<Result> find(List<Result> results, Workload workload, int concurrency, String target) {
        return results.stream()
                .filter(r -> r.workload() == workload && r.concurrency() == concurrency && r.target().equals(target))
                .findFirst();
    }

    private static String json(BenchArgs args, String gpu, Correctness.Report check, List<Result> results) throws IOException {
        ObjectNode root = Target.JSON.createObjectNode();
        root.put("gpu", gpu);
        root.put("java", Runtime.version().toString());
        root.put("seed", args.seed());
        root.put("warmupSeconds", args.warmup().toSeconds());
        root.put("durationSeconds", args.duration().toSeconds());
        ObjectNode correctness = root.putObject("correctness");
        check.minCosine().forEach((t, v) -> correctness.putObject(t).put("minCosine", v));
        check.maxScoreDiff().forEach((t, v) -> correctness.withObject(t).put("maxScoreDiff", v).put("plantedFirst", check.sameTop().get(t)));
        ObjectNode tokens = root.putObject("tokensPerInput");
        check.tokensPerInput().forEach((w, v) -> tokens.put(w.label(), v));
        ArrayNode runs = root.putArray("runs");
        for (Result r : results) {
            ObjectNode run = runs.addObject()
                    .put("workload", r.workload().label())
                    .put("target", r.target())
                    .put("concurrency", r.concurrency())
                    .put("requests", r.requests())
                    .put("errors", r.errors())
                    .put("seconds", r.seconds())
                    .put("requestsPerSecond", r.requestsPerSecond())
                    .put("inputsPerSecond", r.inputsPerSecond());
            putFinite(run, Map.of(
                    "tokensPerSecond", r.tokensPerSecond(),
                    "p50Ms", r.p50(), "p90Ms", r.p90(), "p99Ms", r.p99(), "maxMs", r.max(),
                    "gpuUtilization", r.gpuUtilization(),
                    "allocatedBytesPerRequest", r.allocatedPerRequest(),
                    "gcPauseMs", r.gcPauseMs(),
                    "paddingRatio", r.paddingRatio()));
            if (r.gpuMemoryMib() >= 0) {
                run.put("gpuMemoryMib", r.gpuMemoryMib());
            }
            if (r.firstError() != null) {
                run.put("firstError", r.firstError());
            }
        }
        return Target.JSON.writerWithDefaultPrettyPrinter().writeValueAsString(root);
    }

    private static void putFinite(ObjectNode node, Map<String, Double> values) {
        values.forEach((key, value) -> {
            if (Double.isFinite(value)) {
                node.put(key, value);
            }
        });
    }

    private static String f0(double v) {
        return Double.isFinite(v) ? String.format(Locale.ROOT, "%.0f", v) : "-";
    }

    private static String f1(double v) {
        return Double.isFinite(v) ? String.format(Locale.ROOT, "%.1f", v) : "-";
    }

    private static String bytes(double v) {
        if (!Double.isFinite(v)) {
            return "-";
        }
        if (v >= 1 << 20) {
            return String.format(Locale.ROOT, "%.1f MiB", v / (1 << 20));
        }
        return String.format(Locale.ROOT, "%.0f KiB", v / 1024);
    }
}

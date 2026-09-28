package com.ragulabs.djembed.bench;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Benchmarks Djembed against Text Embeddings Inference on identical requests.
 *
 * <p>For each workload and concurrency level the targets run one after the other, so only one server is under load
 * at a time and slow drifts (clocks, temperature) spread evenly instead of favouring whoever ran first.
 */
public final class Bench {

    /** Distinct requests per workload; workers cycle through them. */
    private static final int POOL = 256;

    private Bench() {
    }

    public static void main(String[] argv) throws Exception {
        BenchArgs args;
        try {
            args = BenchArgs.parse(argv);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage() + "\n\n" + BenchArgs.USAGE);
            System.exit(2);
            return;
        }

        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(10))
                .executor(Executors.newVirtualThreadPerTaskExecutor())
                .build();
        List<Target> targets = args.targets();
        String gpu = args.gpu() == null ? null : GpuSampler.describe(args.gpu());
        System.out.println("Targets: " + targets.stream().map(Target::name).toList() + (gpu == null ? "" : ", " + gpu));

        Correctness.Report check = Correctness.check(client, targets, args.seed());
        check.minCosine().forEach((t, v) -> System.out.printf(Locale.ROOT, "Embeddings, min cosine vs reference: %s %.6f%n", t, v));
        check.maxScoreDiff().forEach((t, v) -> System.out.printf(Locale.ROOT,
                "Rerank, max score difference vs reference: %s %.2e, planted document first: %s%n", t, v, check.sameTop().get(t)));
        check.problems().forEach(p -> System.out.println("Problem: " + p));

        List<Result> results = new ArrayList<>();
        for (Workload workload : args.workloads()) {
            Map<Target, List<HttpRequest>> pools = new LinkedHashMap<>();
            List<Corpus.Payload> payloads = Corpus.payloads(workload, args.seed(), POOL);
            for (Target target : targets) {
                if (target.supports(workload)) {
                    pools.put(target, payloads.stream()
                            .map(p -> workload.isEmbedding() ? target.embed(p.texts()) : target.rerank(p.query(), p.texts()))
                            .toList());
                }
            }
            if (pools.isEmpty()) {
                continue;
            }
            double tokensPerInput = check.tokensPerInput().getOrDefault(workload, Double.NaN);
            Report.header(workload, tokensPerInput);
            String model = workload.isEmbedding() ? args.embedModel() : args.rerankModel();
            for (int concurrency : args.concurrency()) {
                for (Map.Entry<Target, List<HttpRequest>> e : pools.entrySet()) {
                    Result result = measure(client, args, e.getKey(), e.getValue(), workload, concurrency, model, tokensPerInput);
                    results.add(result);
                    Report.row(result);
                }
            }
        }

        Report.summary(args, results);
        String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        Report.write(args.out().resolve(stamp), args, gpu, check, results);
    }

    private static Result measure(HttpClient client, BenchArgs args, Target target, List<HttpRequest> pool, Workload workload,
                                  int concurrency, String model, double tokensPerInput) throws InterruptedException {
        GpuSampler sampler = args.gpu() == null ? null : new GpuSampler(args.gpu());
        Prometheus[] scrapes = new Prometheus[2];
        GpuSampler.Sample[] gpu = new GpuSampler.Sample[1];
        boolean[] sampling = new boolean[1];

        LoadRunner.Measurement m = LoadRunner.run(client, pool, concurrency, args.warmup(), args.duration(), new LoadRunner.Window() {
            @Override
            public void opened() {
                scrapes[0] = scrape(client, target);
                if (sampler != null) {
                    try {
                        sampler.start();
                        sampling[0] = true;
                    } catch (IOException e) {
                        System.out.println("nvidia-smi unavailable: " + e.getMessage());
                    }
                }
            }

            @Override
            public void closed() {
                scrapes[1] = scrape(client, target);
                if (sampling[0]) {
                    try {
                        gpu[0] = sampler.stop();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        });

        double allocated = Double.NaN;
        double gcPauseMs = Double.NaN;
        double padding = Double.NaN;
        if (scrapes[0] != null && scrapes[1] != null) {
            allocated = (scrapes[1].sum("djembed_jvm_allocated_bytes_total") - scrapes[0].sum("djembed_jvm_allocated_bytes_total"))
                    / Math.max(1, m.requests());
            gcPauseMs = (scrapes[1].sum("jvm_gc_pause_seconds_sum") - scrapes[0].sum("jvm_gc_pause_seconds_sum")) * 1000;
            String label = "model=\"" + model + "\"";
            double real = scrapes[1].sum("djembed_tokens_total", label, "kind=\"real\"") - scrapes[0].sum("djembed_tokens_total", label, "kind=\"real\"");
            double pad = scrapes[1].sum("djembed_tokens_total", label, "kind=\"padding\"") - scrapes[0].sum("djembed_tokens_total", label, "kind=\"padding\"");
            padding = real + pad > 0 ? pad / (real + pad) : Double.NaN;
        }
        double[] latency = m.requests() == 0 ? new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN} : new double[]{
                m.latency().getValueAtPercentile(50) / 1000.0,
                m.latency().getValueAtPercentile(90) / 1000.0,
                m.latency().getValueAtPercentile(99) / 1000.0,
                m.latency().getMaxValue() / 1000.0};
        return new Result(workload, target.name(), concurrency, m.requests(), m.errors(), m.seconds(), latency, m.firstError(),
                tokensPerInput,
                gpu[0] == null ? Double.NaN : gpu[0].utilization(),
                gpu[0] == null ? -1 : gpu[0].maxMemoryMib(),
                allocated, gcPauseMs, padding);
    }

    private static Prometheus scrape(HttpClient client, Target target) {
        if (target.metrics() == null) {
            return null;
        }
        try {
            return Prometheus.scrape(client, target.metrics());
        } catch (IOException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }
}

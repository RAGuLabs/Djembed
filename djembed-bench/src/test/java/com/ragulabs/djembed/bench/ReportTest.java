package com.ragulabs.djembed.bench;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReportTest {

    private static final BenchArgs ARGS = new BenchArgs(null, null, null, null, "e", "r", List.of(Workload.INGEST),
            List.of(8, 32), Duration.ofSeconds(1), Duration.ofSeconds(10), 42, Path.of("out"), null);

    @Test
    void runsWithErrorsAreNotCompared() {
        List<Result> results = List.of(
                result("djembed", 8, 40, 0), result("tei", 8, 20, 0),
                result("djembed", 32, 40, 0), result("tei", 32, 20, 5));

        String md = Report.markdown(ARGS, null, emptyCheck(), results);

        assertTrue(md.contains("| ingest | 8 | 2.00× | 0.50× |"), md);
        assertTrue(md.contains("| ingest | 32 | invalid: errors | invalid: errors |"), md);
    }

    @Test
    void correctnessFailuresNameTheReason() {
        Correctness.Report check = new Correctness.Report(
                Map.of("djembed", 1.0, "tei", 0.97), Map.of("tei", 0.2), Map.of("tei", false), Map.of(), List.of("tei embed: timeout"));

        List<String> failures = check.failures();

        assertEquals(4, failures.size(), failures.toString());
        assertTrue(failures.stream().anyMatch(f -> f.contains("min cosine 0.970000")));
        assertTrue(failures.stream().anyMatch(f -> f.contains("planted")));
    }

    private static Result result(String target, int concurrency, long requests, long errors) {
        // Latency scales inversely with throughput: twice the requests, half the p99.
        double p99 = 1000.0 / requests;
        return new Result(Workload.INGEST, target, concurrency, requests, errors, 10, new double[]{p99, p99, p99, p99}, null,
                Double.NaN, Double.NaN, -1, Double.NaN, Double.NaN, Double.NaN);
    }

    private static Correctness.Report emptyCheck() {
        return new Correctness.Report(Map.of(), Map.of(), Map.of(), Map.of(), List.of());
    }
}

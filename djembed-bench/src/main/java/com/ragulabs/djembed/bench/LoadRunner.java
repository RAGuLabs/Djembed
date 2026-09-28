package com.ragulabs.djembed.bench;

import org.HdrHistogram.Histogram;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * Closed-loop load: {@code concurrency} workers each send a request, wait for the answer and send the next.
 *
 * <p>Workers are virtual threads. This is the textbook case for them: each spends nearly all its time blocked on a
 * socket, so thousands cost almost nothing, and the load generator stays out of the way of the servers it measures.
 * Every worker records into its own histogram, merged at the end, so measuring adds no contention.
 *
 * <p>After the warm-up, only requests completing inside the measured window count; a failed request is followed by
 * a short back-off. Response bodies are drained but
 * not parsed, keeping client CPU out of the numbers (content is checked separately, before any load).
 */
final class LoadRunner {

    /** Latencies are recorded in microseconds, up to ten minutes, to three significant digits. */
    private static final long MAX_LATENCY_MICROS = TimeUnit.MINUTES.toMicros(10);

    /**
     * Pause after a failed request. Without it a refusing server is hit by a retry storm that measures how fast it
     * can say no and starves the requests it does accept.
     */
    static final Duration ERROR_BACKOFF = Duration.ofMillis(100);

    record Measurement(long requests, long errors, double seconds, Histogram latency, String firstError) {

        double requestsPerSecond() {
            return requests / seconds;
        }
    }

    /** Hooks run exactly at the start and end of the measured window, e.g. to scrape server metrics. */
    interface Window {
        void opened();

        void closed();
    }

    private LoadRunner() {
    }

    static Measurement run(HttpClient client, List<HttpRequest> pool, int concurrency, Duration warmup, Duration duration,
                           Window window) throws InterruptedException {
        long start = System.nanoTime();
        long open = start + warmup.toNanos();
        long close = open + duration.toNanos();
        AtomicLong cursor = new AtomicLong();
        AtomicLong errors = new AtomicLong();
        List<String> firstError = new ArrayList<>(1);

        List<Histogram> histograms = new ArrayList<>(concurrency);
        List<Thread> workers = new ArrayList<>(concurrency);
        for (int w = 0; w < concurrency; w++) {
            Histogram histogram = new Histogram(MAX_LATENCY_MICROS, 3);
            histograms.add(histogram);
            workers.add(Thread.ofVirtual().name("bench-worker-" + w).start(() -> {
                while (System.nanoTime() < close) {
                    HttpRequest request = pool.get((int) (cursor.getAndIncrement() % pool.size()));
                    long sent = System.nanoTime();
                    String failure;
                    try {
                        HttpResponse<Void> response = client.send(request, HttpResponse.BodyHandlers.discarding());
                        failure = response.statusCode() == 200 ? null : "HTTP " + response.statusCode();
                    } catch (IOException e) {
                        failure = e.toString();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                    long received = System.nanoTime();
                    if (failure != null) {
                        LockSupport.parkNanos(ERROR_BACKOFF.toNanos());
                    }
                    if (received < open || received > close) {
                        continue;
                    }
                    if (failure == null) {
                        histogram.recordValue(Math.min(MAX_LATENCY_MICROS, TimeUnit.NANOSECONDS.toMicros(received - sent)));
                    } else {
                        errors.incrementAndGet();
                        synchronized (firstError) {
                            if (firstError.isEmpty()) {
                                firstError.add(failure);
                            }
                        }
                    }
                }
            }));
        }

        LockSupport.parkNanos(open - System.nanoTime());
        window.opened();
        LockSupport.parkNanos(close - System.nanoTime());
        window.closed();
        for (Thread worker : workers) {
            worker.join();
        }

        Histogram latency = new Histogram(MAX_LATENCY_MICROS, 3);
        histograms.forEach(latency::add);
        return new Measurement(latency.getTotalCount(), errors.get(), duration.toNanos() / 1e9, latency,
                firstError.isEmpty() ? null : firstError.getFirst());
    }
}

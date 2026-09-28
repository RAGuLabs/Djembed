package com.ragulabs.djembed.server.metrics;

import com.linecorp.armeria.server.HttpService;
import com.linecorp.armeria.server.prometheus.PrometheusExpositionService;
import com.sun.management.ThreadMXBean;
import com.ragulabs.djembed.core.EngineObserver;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.binder.jvm.ClassLoaderMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmGcMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmMemoryMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmThreadMetrics;
import io.micrometer.core.instrument.binder.system.ProcessorMetrics;
import io.micrometer.prometheusmetrics.PrometheusConfig;
import io.micrometer.prometheusmetrics.PrometheusMeterRegistry;

import java.lang.management.ManagementFactory;
import java.util.concurrent.TimeUnit;
import java.util.function.IntSupplier;

/**
 * Prometheus metrics: per-model device activity (forward passes, batch shape, real versus padding tokens, queue
 * depth), plus JVM memory and GC, which is where allocation pressure shows.
 */
public final class DjembedMetrics {

    private static final String MODEL = "model";

    private final PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);

    public DjembedMetrics() {
        new JvmMemoryMetrics().bindTo(registry);
        new JvmGcMetrics().bindTo(registry);
        new JvmThreadMetrics().bindTo(registry);
        new ClassLoaderMetrics().bindTo(registry);
        new ProcessorMetrics().bindTo(registry);
        bindAllocatedBytes();
    }

    /**
     * Heap bytes allocated since start, from the JVM's per-thread allocation accounting. Micrometer's
     * {@code jvm.gc.memory.allocated} only moves when a collection happens, so at a low allocation rate it can stay
     * flat for minutes; this counter is exact at every scrape.
     */
    private void bindAllocatedBytes() {
        if (ManagementFactory.getThreadMXBean() instanceof ThreadMXBean threads && threads.isThreadAllocatedMemorySupported()) {
            FunctionCounter.builder("djembed.jvm.allocated", threads, ThreadMXBean::getTotalThreadAllocatedBytes)
                    .description("Heap bytes allocated by all threads since the JVM started")
                    .baseUnit("bytes")
                    .register(registry);
        }
    }

    public MeterRegistry registry() {
        return registry;
    }

    public HttpService exposition() {
        return PrometheusExpositionService.of(registry.getPrometheusRegistry());
    }

    /** Observer recording the forward passes of {@code model}. */
    public EngineObserver observer(String model) {
        Timer passes = Timer.builder("djembed.forward.pass")
                .description("Forward passes on the device, input transfer and output copy included")
                .tag(MODEL, model)
                .publishPercentileHistogram()
                .register(registry);
        DistributionSummary rows = DistributionSummary.builder("djembed.batch.rows")
                .description("Sequences per forward pass")
                .tag(MODEL, model)
                .register(registry);
        Counter real = Counter.builder("djembed.tokens")
                .description("Token slots computed by the device, by whether they held a real token or padding")
                .tags(MODEL, model, "kind", "real")
                .register(registry);
        Counter padding = Counter.builder("djembed.tokens")
                .description("Token slots computed by the device, by whether they held a real token or padding")
                .tags(MODEL, model, "kind", "padding")
                .register(registry);
        return new EngineObserver() {
            @Override
            public void forwardPass(int batchRows, int rowLength, long tokens, long nanos) {
                passes.record(nanos, TimeUnit.NANOSECONDS);
                rows.record(batchRows);
                real.increment(tokens);
                padding.increment((double) batchRows * rowLength - tokens);
            }
        };
    }

    /** Gauge of the inputs {@code model} has accepted and not yet completed. */
    public void queue(String model, IntSupplier queuedInputs) {
        Gauge.builder("djembed.queue.inputs", queuedInputs::getAsInt)
                .description("Inputs accepted and not yet completed")
                .tag(MODEL, model)
                .register(registry);
    }
}

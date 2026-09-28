package com.ragulabs.djembed.bench;

/**
 * One measured run: a target under a workload at a concurrency level. Unknown values are {@code NaN}.
 *
 * @param latencyMs    p50, p90, p99 and max latency in milliseconds
 * @param allocatedPerRequest heap bytes the server JVM allocated per request, all threads (Djembed only)
 * @param gcPauseMs    GC pause time of the server JVM during the window (Djembed only)
 * @param paddingRatio share of computed token slots that were padding (Djembed only)
 */
record Result(
        Workload workload,
        String target,
        int concurrency,
        long requests,
        long errors,
        double seconds,
        double[] latencyMs,
        String firstError,
        double tokensPerInput,
        double gpuUtilization,
        long gpuMemoryMib,
        double allocatedPerRequest,
        double gcPauseMs,
        double paddingRatio) {

    double requestsPerSecond() {
        return requests / seconds;
    }

    double inputsPerSecond() {
        return requestsPerSecond() * workload.inputs;
    }

    double tokensPerSecond() {
        return inputsPerSecond() * tokensPerInput;
    }

    double p50() {
        return latencyMs[0];
    }

    double p90() {
        return latencyMs[1];
    }

    double p99() {
        return latencyMs[2];
    }

    double max() {
        return latencyMs[3];
    }
}

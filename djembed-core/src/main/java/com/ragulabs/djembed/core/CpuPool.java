package com.ragulabs.djembed.core;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The process-wide pool for CPU-bound work around inference: tokenization, completing callers' futures (so their
 * callbacks never run on a device thread), and whatever CPU work callers chain on those futures. One daemon platform
 * thread per core: the work is CPU-bound and partly native (JNI), where virtual threads would pin their carriers
 * and bring nothing but oversubscription.
 */
public final class CpuPool {

    private static final ExecutorService POOL = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors(),
            Thread.ofPlatform().name("djembed-cpu-", 0).daemon(true).factory());

    private CpuPool() {
    }

    public static ExecutorService executor() {
        return POOL;
    }
}

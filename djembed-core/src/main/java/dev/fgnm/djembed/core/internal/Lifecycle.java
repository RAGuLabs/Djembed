package dev.fgnm.djembed.core.internal;

import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Guards native resources (tokenizer, session) against use after release: work runs under a shared lock,
 * closing takes the exclusive one and so waits for work in progress.
 */
public final class Lifecycle {

    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private boolean closed;

    /** Runs {@code work} unless closed; returns whether it ran. */
    public boolean runIfOpen(Runnable work) {
        lock.readLock().lock();
        try {
            if (closed) {
                return false;
            }
            work.run();
            return true;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Marks closed once work in progress has finished; false if it already was. */
    public boolean close() {
        lock.writeLock().lock();
        try {
            if (closed) {
                return false;
            }
            closed = true;
            return true;
        } finally {
            lock.writeLock().unlock();
        }
    }
}

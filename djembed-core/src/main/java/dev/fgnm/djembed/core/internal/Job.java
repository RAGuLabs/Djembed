package dev.fgnm.djembed.core.internal;

import java.util.concurrent.CompletableFuture;

/**
 * One tokenized request inside a {@link Scheduler}: its sequences, which of them are still to be scheduled, and the
 * future its caller waits on. Scheduling state is touched only by the scheduler thread.
 *
 * @param <R> the result type
 */
public abstract class Job<R> {

    private final CompletableFuture<R> future;
    private final int[] lengths;
    private final int[] order;
    private int next;
    private int remaining;

    /**
     * @param future  the caller's future, created before tokenization so a cancellation made meanwhile still
     *                reaches the scheduler
     * @param lengths sequence lengths, in tokens, special tokens included
     * @param count   number of sequences
     */
    protected Job(CompletableFuture<R> future, int[] lengths, int count) {
        this.future = future;
        this.lengths = lengths;
        // Each round takes a job's shortest pending sequences first, so what it contributes packs tightly.
        this.order = BatchPlanner.sortedByLength(lengths, count);
        this.remaining = count;
    }

    public CompletableFuture<R> future() {
        return future;
    }

    boolean hasUnscheduled() {
        return next < order.length;
    }

    int nextLength() {
        return lengths[order[next]];
    }

    /** Takes the next sequence to schedule. */
    int take() {
        return order[next++];
    }

    /** Records one finished sequence; true when it was the last. */
    boolean finishOne() {
        return --remaining == 0;
    }

    int length(int sequence) {
        return lengths[sequence];
    }
}

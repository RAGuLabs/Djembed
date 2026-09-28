package com.ragulabs.djembed.core.internal;

/**
 * The model-specific half of a {@link Scheduler}: how sequences become tensor rows and how outputs become results.
 * Called only from the scheduler thread, one forward pass at a time.
 *
 * @param <R> the result type
 * @param <J> the job type
 */
public interface BatchRunner<R, J extends Job<R>> {

    /** Starts a forward pass of {@code rows} sequences padded to {@code rowLength} tokens. */
    void begin(int rows, int rowLength);

    void putRow(int row, J job, int sequence);

    void run();

    /** Stores the output of {@code row} into the job it came from. */
    void readRow(int row, J job, int sequence);

    /** Builds the result of a job whose sequences have all been read. */
    R finish(J job);
}

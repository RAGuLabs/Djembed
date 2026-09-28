package com.ragulabs.djembed.core.internal;

import com.ragulabs.djembed.core.CpuPool;
import com.ragulabs.djembed.core.EngineObserver;
import com.ragulabs.djembed.core.EngineOverloadedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Runs the forward passes of one model on a dedicated thread, batching sequences across requests.
 *
 * <p>While a round runs, newly tokenized jobs queue up; the next round takes them together with whatever earlier
 * jobs still have pending (continuous batching: no waiting window, so a lone request pays no extra latency). A round
 * holds up to {@value #ROUND_BATCHES} full batches of tokens, split fairly: every pending job gets an equal share,
 * starting from its shortest sequences, so a query arriving during a large ingestion request is served in the next
 * round instead of after the whole ingestion. The round's sequences are then sorted and packed by
 * {@link BatchPlanner}, and a job completes as soon as its last sequence is read.
 *
 * <p>Admission is bounded by queued inputs: past the limit, new requests fail fast with
 * {@link EngineOverloadedException} rather than growing the queue without bound. Cancelled jobs are dropped before
 * their remaining sequences reach the device.
 */
public final class Scheduler<R, J extends Job<R>> implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Scheduler.class);

    /** Round size in full batches: enough for length sorting to pay off, small enough to keep newcomers waiting briefly. */
    static final int ROUND_BATCHES = 4;

    private final String name;
    private final BatchRunner<R, J> runner;
    private final int maxRows;
    private final long tokenBudget;
    private final int maxQueuedInputs;
    private final Executor completions;
    private final EngineObserver observer;

    private final LinkedBlockingQueue<J> incoming = new LinkedBlockingQueue<>();
    private final AtomicInteger queuedInputs = new AtomicInteger();
    private final Thread thread;
    private volatile boolean closed;

    // Scheduler-thread state, reused across rounds.
    private final ArrayDeque<J> active = new ArrayDeque<>();
    private final List<J> roundJobs = new ArrayList<>();
    private int[] roundSequences = new int[64];
    private int[] roundLengths = new int[64];
    private final BatchPlanner planner = new BatchPlanner();

    public Scheduler(String name, BatchRunner<R, J> runner, int maxRows, long tokenBudget, int maxQueuedInputs,
                     Executor completions, EngineObserver observer) {
        this.name = name;
        this.runner = runner;
        this.maxRows = maxRows;
        this.tokenBudget = tokenBudget;
        this.maxQueuedInputs = maxQueuedInputs;
        this.completions = completions;
        this.observer = observer;
        // A platform thread: it lives as long as the engine and spends its time inside native forward passes,
        // which would pin a virtual thread to its carrier anyway.
        this.thread = Thread.ofPlatform().name("djembed-scheduler-" + name).daemon(true).unstarted(this::loop);
        thread.start();
    }

    /**
     * Reserves room for {@code inputs} caller inputs before any work is spent on them, held until {@code future}
     * completes however it does (result, failure, cancellation). A lone request is always admitted, whatever its
     * size, so no request is unservable.
     *
     * @throws EngineOverloadedException when the queue is full
     */
    public void admit(int inputs, CompletableFuture<?> future) {
        if (closed) {
            throw new IllegalStateException("Engine '" + name + "' is closed");
        }
        int queued = queuedInputs.addAndGet(inputs);
        if (queued > maxQueuedInputs && queued != inputs) {
            queuedInputs.addAndGet(-inputs);
            throw new EngineOverloadedException("Model '" + name + "' is overloaded: " + (queued - inputs)
                    + " inputs queued, limit " + maxQueuedInputs);
        }
        future.whenComplete((result, failure) -> queuedInputs.addAndGet(-inputs));
    }

    /**
     * The whole path of a request: admission, then {@code prepare} (tokenization) on the CPU pool while
     * {@code lifecycle} is open, then the queue. The returned future is the one handed to {@code prepare}.
     *
     * @throws EngineOverloadedException when the queue is full
     */
    public CompletableFuture<R> submitAsync(int inputs, Lifecycle lifecycle, Function<CompletableFuture<R>, J> prepare) {
        CompletableFuture<R> future = new CompletableFuture<>();
        admit(inputs, future);
        CpuPool.executor().execute(() -> {
            if (future.isDone()) {
                return;
            }
            try {
                boolean ran = lifecycle.runIfOpen(() -> submit(prepare.apply(future)));
                if (!ran) {
                    future.completeExceptionally(new IllegalStateException("Engine '" + name + "' is closed"));
                }
            } catch (RuntimeException | Error e) {
                future.completeExceptionally(e);
            }
        });
        return future;
    }

    /** Inputs admitted and not yet completed. */
    public int queuedInputs() {
        return queuedInputs.get();
    }

    /** Hands a job to the scheduler thread; its future completes on the completions executor. */
    public void submit(J job) {
        if (closed) {
            fail(job, new IllegalStateException("Engine '" + name + "' is closed"));
            return;
        }
        incoming.add(job);
        // close() may have drained the queue between the check and the add; make sure the job is not stranded.
        if (closed && incoming.remove(job)) {
            fail(job, new IllegalStateException("Engine '" + name + "' is closed"));
        }
    }

    private void loop() {
        while (!closed) {
            try {
                if (active.isEmpty()) {
                    J first = incoming.poll(1, TimeUnit.SECONDS);
                    if (first == null) {
                        continue;
                    }
                    active.add(first);
                }
                incoming.drainTo(active);
                runRound();
            } catch (InterruptedException e) {
                break;
            } catch (RuntimeException | Error e) {
                log.error("Scheduler '{}' failed a round", name, e);
                failRound(e);
            }
        }
    }

    private void runRound() {
        int count = collectRound();
        if (count == 0) {
            return;
        }
        planner.plan(roundLengths, count, maxRows, tokenBudget);
        for (int b = 0; b < planner.batches(); b++) {
            int first = planner.start(b);
            int rows = planner.rows(b);
            // Ascending order: the batch's last sequence is its longest.
            int rowLength = roundLengths[planner.sequence(first + rows - 1)];
            runner.begin(rows, rowLength);
            long tokens = 0;
            for (int r = 0; r < rows; r++) {
                int slot = planner.sequence(first + r);
                runner.putRow(r, roundJobs.get(slot), roundSequences[slot]);
                tokens += roundLengths[slot];
            }
            long started = System.nanoTime();
            runner.run();
            observer.forwardPass(rows, rowLength, tokens, System.nanoTime() - started);
            for (int r = 0; r < rows; r++) {
                int slot = planner.sequence(first + r);
                J job = roundJobs.get(slot);
                runner.readRow(r, job, roundSequences[slot]);
                if (job.finishOne() && !job.future().isDone()) {
                    complete(job, runner.finish(job));
                }
            }
        }
        roundJobs.clear();
    }

    /** Takes each active job's fair share of the round into the round arrays; returns the number of sequences. */
    private int collectRound() {
        roundJobs.clear();
        long share = Math.max(1, (long) ROUND_BATCHES * tokenBudget / active.size());
        int count = 0;
        for (Iterator<J> it = active.iterator(); it.hasNext(); ) {
            J job = it.next();
            if (job.future().isDone()) {
                it.remove();
                continue;
            }
            long taken = 0;
            while (job.hasUnscheduled() && (taken == 0 || taken + job.nextLength() <= share)) {
                taken += job.nextLength();
                int sequence = job.take();
                ensureRoundCapacity(count + 1);
                roundJobs.add(job);
                roundSequences[count] = sequence;
                roundLengths[count] = job.length(sequence);
                count++;
            }
            if (!job.hasUnscheduled()) {
                it.remove();
            }
        }
        return count;
    }

    private void ensureRoundCapacity(int needed) {
        if (needed > roundSequences.length) {
            int capacity = Math.max(needed, roundSequences.length * 2);
            roundSequences = Arrays.copyOf(roundSequences, capacity);
            roundLengths = Arrays.copyOf(roundLengths, capacity);
        }
    }

    private void failRound(Throwable cause) {
        for (J job : roundJobs) {
            fail(job, cause);
        }
        roundJobs.clear();
    }

    private void complete(J job, R result) {
        completions.execute(() -> job.future().complete(result));
    }

    private void fail(J job, Throwable cause) {
        completions.execute(() -> job.future().completeExceptionally(cause));
    }

    /** Stops the thread after the forward pass in progress and fails every job not yet completed. */
    @Override
    public void close() {
        closed = true;
        thread.interrupt();
        try {
            thread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        IllegalStateException cause = new IllegalStateException("Engine '" + name + "' is closed");
        for (J job : active) {
            fail(job, cause);
        }
        for (J job : roundJobs) {
            fail(job, cause);
        }
        J job;
        while ((job = incoming.poll()) != null) {
            fail(job, cause);
        }
    }
}

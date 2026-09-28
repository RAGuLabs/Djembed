package com.ragulabs.djembed.core.internal;

import com.ragulabs.djembed.core.EngineObserver;
import com.ragulabs.djembed.core.EngineOverloadedException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Timeout(10)
class SchedulerTest {

    private static final int TOKEN_BUDGET = 10;

    private final FakeRunner runner = new FakeRunner();
    private Scheduler<int[], FakeJob> scheduler;

    @AfterEach
    void close() {
        runner.gate.countDown();
        if (scheduler != null) {
            scheduler.close();
        }
    }

    @Test
    void jobsArrivingWhileTheDeviceIsBusyShareABatch() throws Exception {
        scheduler = scheduler(64, 100);
        runner.blockFirstRun();
        FakeJob a = submit("a", 1, 1);
        runner.firstRunStarted.await();

        FakeJob b = submit("b", 1, 1);
        FakeJob c = submit("c", 1, 1);
        runner.gate.countDown();

        a.future().get(5, TimeUnit.SECONDS);
        b.future().get(5, TimeUnit.SECONDS);
        c.future().get(5, TimeUnit.SECONDS);
        assertEquals(List.of(List.of("a0"), List.of("b0", "c0")), runner.batches);
    }

    @Test
    void aSmallJobIsNotStuckBehindALargeOne() throws Exception {
        scheduler = scheduler(64, 1_000);
        runner.blockFirstRun();
        FakeJob large = submit("large", 100);
        runner.firstRunStarted.await();

        FakeJob small = submit("small", 1);
        runner.gate.countDown();

        small.future().get(5, TimeUnit.SECONDS);
        large.future().get(5, TimeUnit.SECONDS);
        // Round 1 took 4 batches' worth of the large job alone. Round 2 split its 40 tokens between two jobs,
        // so the small one completed there, while most of the large one was still pending.
        int smallDone = runner.completions.indexOf("small");
        assertTrue(smallDone >= 0 && smallDone < runner.completions.indexOf("large"));
        assertTrue(runner.rowsBefore("small0") < 20, "small ran after " + runner.rowsBefore("small0") + " rows");
    }

    @Test
    void cancelledJobsAreDropped() throws Exception {
        scheduler = scheduler(64, 1_000);
        runner.blockFirstRun();
        FakeJob first = submit("first", 1);
        runner.firstRunStarted.await();

        FakeJob cancelled = submit("cancelled", 50);
        cancelled.future().cancel(false);
        FakeJob after = submit("after", 1);
        runner.gate.countDown();

        first.future().get(5, TimeUnit.SECONDS);
        after.future().get(5, TimeUnit.SECONDS);
        assertEquals(-1, runner.rowsBefore("cancelled0"));
    }

    @Test
    void aFailedForwardPassFailsItsJobsOnly() throws Exception {
        scheduler = scheduler(64, 100);
        runner.failNextRun = true;
        FakeJob failed = submit("failed", 1);

        CompletionException e = assertThrows(CompletionException.class, () -> failed.future().join());
        assertEquals("device error", e.getCause().getMessage());

        FakeJob next = submit("next", 2);
        assertArrayEquals(new int[]{0, 1}, sorted(next.future().get(5, TimeUnit.SECONDS)));
    }

    @Test
    void admissionIsBoundedButNeverRefusesAnIdleEngine() {
        scheduler = scheduler(64, 2);
        CompletableFuture<int[]> big = new CompletableFuture<>();
        scheduler.admit(5, big);

        assertThrows(EngineOverloadedException.class, () -> scheduler.admit(1, new CompletableFuture<>()));

        big.complete(new int[0]);
        CompletableFuture<int[]> next = new CompletableFuture<>();
        scheduler.admit(2, next);
        assertEquals(2, scheduler.queuedInputs());
        next.cancel(false);
        assertEquals(0, scheduler.queuedInputs());
    }

    @Test
    void closeFailsWhatIsPending() throws Exception {
        scheduler = scheduler(64, 100);
        runner.blockFirstRun();
        submit("running", 1);
        runner.firstRunStarted.await();
        FakeJob waiting = submit("waiting", 1);

        // close() interrupts the scheduler thread, which releases the held forward pass.
        scheduler.close();

        CompletionException e = assertThrows(CompletionException.class, () -> waiting.future().join());
        assertInstanceOf(IllegalStateException.class, e.getCause());
        assertFalse(runner.batches.stream().flatMap(List::stream).anyMatch("waiting0"::equals));
    }

    private Scheduler<int[], FakeJob> scheduler(int maxRows, int maxQueuedInputs) {
        return new Scheduler<>("test", runner, new BatchLimits(maxRows, TOKEN_BUDGET, false, TOKEN_BUDGET), maxQueuedInputs,
                Runnable::run, EngineObserver.NONE);
    }

    /** A job of {@code sequences} sequences filling a whole batch each, named {@code name0, name1, …}. */
    private FakeJob submit(String name, int sequences) {
        return submit(name, sequences, TOKEN_BUDGET);
    }

    private FakeJob submit(String name, int sequences, int length) {
        CompletableFuture<int[]> future = new CompletableFuture<>();
        scheduler.admit(sequences, future);
        FakeJob job = new FakeJob(future, name, sequences, length);
        scheduler.submit(job);
        return job;
    }

    private static int[] sorted(int[] values) {
        int[] copy = values.clone();
        Arrays.sort(copy);
        return copy;
    }

    static final class FakeJob extends Job<int[]> {

        final String name;
        final int[] read;
        int reads;

        FakeJob(CompletableFuture<int[]> future, String name, int sequences, int length) {
            super(future, filled(sequences, length), sequences);
            this.name = name;
            this.read = new int[sequences];
        }

        private static int[] filled(int count, int length) {
            int[] lengths = new int[count];
            Arrays.fill(lengths, length);
            return lengths;
        }
    }

    /** Records every batch as the {@code name + sequence} of its rows; can hold the first run and fail one. */
    static final class FakeRunner implements BatchRunner<int[], FakeJob> {

        final List<List<String>> batches = new CopyOnWriteArrayList<>();
        final List<String> completions = new CopyOnWriteArrayList<>();
        final CountDownLatch firstRunStarted = new CountDownLatch(1);
        CountDownLatch gate = new CountDownLatch(0);
        volatile boolean failNextRun;
        private List<String> current;

        void blockFirstRun() {
            gate = new CountDownLatch(1);
        }

        @Override
        public void begin(int rows, int rowLength) {
            current = new ArrayList<>(rows);
        }

        @Override
        public void putRow(int row, FakeJob job, int sequence) {
            current.add(job.name + sequence);
        }

        @Override
        public void run() {
            batches.add(current);
            firstRunStarted.countDown();
            try {
                gate.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (failNextRun) {
                failNextRun = false;
                throw new IllegalStateException("device error");
            }
        }

        @Override
        public void readRow(int row, FakeJob job, int sequence) {
            job.read[job.reads++] = sequence;
        }

        @Override
        public int[] finish(FakeJob job) {
            completions.add(job.name);
            return job.read;
        }

        /** Rows run before the row named {@code row}, or -1 if it never ran. */
        int rowsBefore(String row) {
            int rows = 0;
            for (List<String> batch : batches) {
                int at = batch.indexOf(row);
                if (at >= 0) {
                    return rows + at;
                }
                rows += batch.size();
            }
            return -1;
        }
    }
}

package dev.fgnm.djembed.core.internal;

import java.util.Arrays;

/**
 * Groups sequences into forward passes.
 *
 * <p>Every row of a batch is padded to the batch's longest sequence, so which sequences share a batch decides how
 * much padding the model computes on. Sequences are sorted by length and packed greedily while
 * {@code rows × longest ≤ tokenBudget} and {@code rows ≤ maxRows}: short inputs travel in wide batches, long ones in
 * narrow batches, and the padded tensor never exceeds the budget. A sequence longer than the budget on its own
 * still gets a batch of one.
 *
 * <p>An instance keeps its working arrays between calls, so planning in steady state allocates nothing. Not
 * thread-safe.
 */
public final class BatchPlanner {

    private long[] keys = new long[0];
    private int[] order = new int[0];
    private int[] starts = new int[1];
    private int batches;

    /** Plans {@code lengths[0 .. count)}; the result stays valid until the next call. */
    public void plan(int[] lengths, int count, int maxRows, long tokenBudget) {
        if (keys.length < count) {
            int capacity = Math.max(count, keys.length * 2);
            keys = new long[capacity];
            order = new int[capacity];
            starts = new int[capacity + 1];
        }
        sortByLength(lengths, count, keys, order);

        batches = 0;
        int from = 0;
        while (from < count) {
            int to = from + 1;
            while (to < count
                    && to - from < maxRows
                    // ascending order: the candidate becomes the batch's longest
                    && (long) (to - from + 1) * lengths[order[to]] <= tokenBudget) {
                to++;
            }
            starts[batches++] = from;
            from = to;
        }
        starts[batches] = count;
    }

    public int batches() {
        return batches;
    }

    /** Position in {@link #sequence} order where batch {@code batch} starts. */
    public int start(int batch) {
        return starts[batch];
    }

    public int rows(int batch) {
        return starts[batch + 1] - starts[batch];
    }

    /** The sequence at {@code position} of the length-ascending order. */
    public int sequence(int position) {
        return order[position];
    }

    /** Indices of {@code lengths[0 .. count)} by ascending length, ties in index order. */
    public static int[] sortedByLength(int[] lengths, int count) {
        int[] sorted = new int[count];
        sortByLength(lengths, count, new long[count], sorted);
        return sorted;
    }

    /** Sorts on primitive keys ({@code length << 32 | index}): no boxing, no comparator. */
    private static void sortByLength(int[] lengths, int count, long[] keys, int[] order) {
        for (int i = 0; i < count; i++) {
            keys[i] = ((long) lengths[i] << 32) | i;
        }
        Arrays.sort(keys, 0, count);
        for (int i = 0; i < count; i++) {
            order[i] = (int) keys[i];
        }
    }
}

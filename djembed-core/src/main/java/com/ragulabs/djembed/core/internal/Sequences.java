package com.ragulabs.djembed.core.internal;

import java.util.Arrays;

/**
 * The model-sized sequences a request expands to, as ranges over the tokenizer output rather than copies of it.
 *
 * <p>Sequence {@code s} is {@code ids[source(s)][from(s) .. to(s))}, wrapped in the tokenizer's single-sequence
 * special tokens when {@link #wrap(int)} is set (a window cut out of a longer encoding lost them). Its
 * {@link #weight(int)} is the number of text tokens it carries, used to average the windows of one input.
 */
public final class Sequences {

    private int size;
    private int[] source;
    private int[] from;
    private int[] to;
    private int[] length;
    private boolean[] wrap;

    private Sequences(int capacity) {
        source = new int[capacity];
        from = new int[capacity];
        to = new int[capacity];
        length = new int[capacity];
        wrap = new boolean[capacity];
    }

    /** One sequence per encoding, taken whole. */
    public static Sequences whole(long[][] ids) {
        Sequences seqs = new Sequences(ids.length);
        for (int i = 0; i < ids.length; i++) {
            seqs.add(i, 0, ids[i].length, false, ids[i].length);
        }
        return seqs;
    }

    /**
     * Encodings that fit {@code maxTokens} are taken whole; longer ones are split into windows of at most
     * {@code maxTokens - lead - trail} text tokens, each ending on a word boundary, and re-wrapped in the
     * {@code lead} / {@code trail} special tokens.
     *
     * @param wordIds per encoding, the word each token belongs to; -1 for special tokens
     */
    public static Sequences chunked(long[][] ids, long[][] wordIds, int maxTokens, int lead, int trail) {
        int window = maxTokens - lead - trail;
        if (window < 1) {
            throw new IllegalArgumentException("maxTokens " + maxTokens + " leaves no room for text around "
                    + (lead + trail) + " special tokens");
        }
        Sequences seqs = new Sequences(ids.length);
        for (int i = 0; i < ids.length; i++) {
            int total = ids[i].length;
            if (total <= maxTokens) {
                seqs.add(i, 0, total, false, total);
                continue;
            }
            long[] words = wordIds[i];
            int start = lead;
            int end = total - trail;
            while (start < end) {
                int stop = Math.min(start + window, end);
                if (stop < end) {
                    stop = wordStartAtOrBefore(words, start, stop);
                }
                seqs.add(i, start, stop, true, lead + (stop - start) + trail);
                start = stop;
            }
        }
        return seqs;
    }

    /**
     * The last index in {@code (start, stop]} where a new word begins, so the window {@code [start, result)} does not
     * cut a word in two. Falls back to {@code stop} when a single word is longer than the whole window.
     */
    static int wordStartAtOrBefore(long[] wordIds, int start, int stop) {
        for (int s = stop; s > start; s--) {
            if (wordIds[s] != wordIds[s - 1]) {
                return s;
            }
        }
        return stop;
    }

    private void add(int src, int f, int t, boolean w, int len) {
        if (size == source.length) {
            int capacity = size * 2;
            source = Arrays.copyOf(source, capacity);
            from = Arrays.copyOf(from, capacity);
            to = Arrays.copyOf(to, capacity);
            length = Arrays.copyOf(length, capacity);
            wrap = Arrays.copyOf(wrap, capacity);
        }
        source[size] = src;
        from[size] = f;
        to[size] = t;
        length[size] = len;
        wrap[size] = w;
        size++;
    }

    public int size() {
        return size;
    }

    public int source(int s) {
        return source[s];
    }

    public int from(int s) {
        return from[s];
    }

    public int to(int s) {
        return to[s];
    }

    public int length(int s) {
        return length[s];
    }

    public boolean wrap(int s) {
        return wrap[s];
    }

    public int weight(int s) {
        return to[s] - from[s];
    }

    /** Backing length array, valid for indices {@code < size()}; for {@link BatchPlanner}. */
    public int[] lengths() {
        return length;
    }
}

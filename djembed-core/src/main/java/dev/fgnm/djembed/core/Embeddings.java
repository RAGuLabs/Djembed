package dev.fgnm.djembed.core;

import java.util.Objects;

/**
 * The vectors of one {@link EmbeddingEngine#embed} call, stored row-major in a single array
 * so a batch costs one allocation instead of one per vector.
 */
public final class Embeddings {

    private final float[] data;
    private final int count;
    private final int dimension;
    private final long promptTokens;
    private final int firstTruncated;

    public Embeddings(float[] data, int count, int dimension, long promptTokens, int firstTruncated) {
        Objects.requireNonNull(data, "data");
        if ((long) count * dimension != data.length) {
            throw new IllegalArgumentException("data holds " + data.length + " floats, expected " + count + " × " + dimension);
        }
        this.data = data;
        this.count = count;
        this.dimension = dimension;
        this.promptTokens = promptTokens;
        this.firstTruncated = firstTruncated;
    }

    public int count() {
        return count;
    }

    public int dimension() {
        return dimension;
    }

    /** Tokens fed to the model, special tokens included. */
    public long promptTokens() {
        return promptTokens;
    }

    /**
     * Index of the first input cut to the token limit under {@link LongInputStrategy#TRUNCATE}, or -1 when every
     * input fitted.
     */
    public int firstTruncated() {
        return firstTruncated;
    }

    public float get(int index, int component) {
        Objects.checkIndex(index, count);
        Objects.checkIndex(component, dimension);
        return data[index * dimension + component];
    }

    /** Copies vector {@code index} into {@code dst} starting at {@code dstOffset}. */
    public void copyTo(int index, float[] dst, int dstOffset) {
        Objects.checkIndex(index, count);
        System.arraycopy(data, index * dimension, dst, dstOffset, dimension);
    }

    /** A fresh copy of vector {@code index}. */
    public float[] vector(int index) {
        float[] v = new float[dimension];
        copyTo(index, v, 0);
        return v;
    }
}

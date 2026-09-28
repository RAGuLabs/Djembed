package com.ragulabs.djembed.core;

import ai.onnxruntime.TensorInfo;
import com.ragulabs.djembed.core.internal.BatchLimits;
import com.ragulabs.djembed.core.internal.BatchRunner;
import com.ragulabs.djembed.core.internal.Encoded;
import com.ragulabs.djembed.core.internal.HfTokenizer;
import com.ragulabs.djembed.core.internal.Job;
import com.ragulabs.djembed.core.internal.Lifecycle;
import com.ragulabs.djembed.core.internal.Limits;
import com.ragulabs.djembed.core.internal.ModelDirectory;
import com.ragulabs.djembed.core.internal.OnnxGraph;
import com.ragulabs.djembed.core.internal.OnnxModel;
import com.ragulabs.djembed.core.internal.Scheduler;
import com.ragulabs.djembed.core.internal.Sequences;
import com.ragulabs.djembed.core.internal.Workspace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * {@link EmbeddingEngine} running an ONNX encoder in-process.
 *
 * <p>Requests are tokenized on a shared CPU pool, then batched across callers by a {@link Scheduler} that owns the
 * device: while one round runs on the device, the next requests are being tokenized.
 */
public final class OnnxEmbeddingEngine implements EmbeddingEngine {

    private static final Logger log = LoggerFactory.getLogger(OnnxEmbeddingEngine.class);

    /** sentence-transformers exports pool (and often normalise) inside the graph under this name. */
    private static final String POOLED_OUTPUT = "sentence_embedding";
    private static final List<String> TOKEN_OUTPUTS = List.of("last_hidden_state", "token_embeddings");

    private final HfTokenizer tokenizer;
    private final OnnxModel model;
    private final Workspace workspace;
    private final Scheduler<Embeddings, EmbedJob> scheduler;
    private final Lifecycle lifecycle = new Lifecycle();

    private final int dimension;
    private final int maxInputTokens;
    private final boolean chunk;
    private final boolean normalize;
    private final PoolingMode pooling;
    private final long[] prefix;
    private final long[] suffix;

    private OnnxEmbeddingEngine(String name, HfTokenizer tokenizer, OnnxModel model, Workspace workspace, int dimension,
                                int maxInputTokens, EmbeddingOptions options, PoolingMode pooling, BatchLimits limits,
                                EngineObserver observer) {
        this.tokenizer = tokenizer;
        this.model = model;
        this.workspace = workspace;
        this.dimension = dimension;
        this.maxInputTokens = maxInputTokens;
        this.chunk = options.longInput() == LongInputStrategy.CHUNK;
        this.normalize = options.normalize();
        this.pooling = pooling;
        this.prefix = tokenizer.prefix();
        this.suffix = tokenizer.suffix();
        EngineOptions engine = options.engine();
        this.scheduler = new Scheduler<>(name, new Runner(), limits, engine.maxQueuedInputs(), CpuPool.executor(), observer);
    }

    /**
     * Loads the model folder at {@code directory} (see the README for the expected layout).
     */
    public static OnnxEmbeddingEngine load(Path directory, EmbeddingOptions options) {
        return load(directory, options, EngineObserver.NONE);
    }

    /**
     * Loads the model folder at {@code directory}, reporting its activity to {@code observer}.
     */
    public static OnnxEmbeddingEngine load(Path directory, EmbeddingOptions options, EngineObserver observer) {
        ModelDirectory dir = ModelDirectory.open(directory);
        EngineOptions engine = options.engine();
        int maxInputTokens = Limits.maxInputTokens(dir, engine);

        HfTokenizer tokenizer = HfTokenizer.load(dir.tokenizerFile(), maxInputTokens, options.longInput() == LongInputStrategy.TRUNCATE);
        OnnxModel model = null;
        Workspace workspace = null;
        try {
            model = OnnxModel.load(dir.onnxFile(), engine.device(), engine.tf32());
            Map<String, TensorInfo> outputs = model.outputs();

            String outputName;
            PoolingMode pooling;
            if (outputs.containsKey(POOLED_OUTPUT)) {
                outputName = POOLED_OUTPUT;
                pooling = null;
            } else {
                outputName = tokenOutput(outputs, directory);
                pooling = options.pooling() != null ? options.pooling() : dir.pooling();
                if (pooling == null) {
                    throw new DjembedException(directory + " outputs token vectors but declares no pooling"
                            + " (no 1_Pooling/config.json): set the pooling option");
                }
            }
            int dimension = width(outputs.get(outputName), dir, outputName);

            if (options.longInput() == LongInputStrategy.CHUNK
                    && maxInputTokens <= tokenizer.prefix().length + tokenizer.suffix().length) {
                throw new DjembedException("maxInputTokens " + maxInputTokens + " leaves no room for text");
            }

            OnnxGraph.Attention attention = OnnxGraph.read(dir.onnxFile()).attention();
            BatchLimits limits = Limits.batchLimits(engine, maxInputTokens, attention == OnnxGraph.Attention.PACKED);
            workspace = new Workspace(model, outputName, dimension,
                    Math.toIntExact(limits.paddedTokens()), maxInputTokens, engine.maxBatchSize(), dir.padTokenId());
            String name = directory.getFileName().toString();
            log.info("Embedding engine {}: dimension={} maxInputTokens={} output={} pooling={} longInput={} attention={}",
                    name, dimension, maxInputTokens, outputName, pooling == null ? "in-graph" : pooling, options.longInput(),
                    attention);
            return new OnnxEmbeddingEngine(name, tokenizer, model, workspace, dimension, maxInputTokens, options, pooling,
                    limits, observer);
        } catch (RuntimeException e) {
            if (workspace != null) {
                workspace.close();
            }
            if (model != null) {
                model.close();
            }
            tokenizer.close();
            throw e;
        }
    }

    private static String tokenOutput(Map<String, TensorInfo> outputs, Path directory) {
        for (String name : TOKEN_OUTPUTS) {
            if (outputs.containsKey(name)) {
                return name;
            }
        }
        for (Map.Entry<String, TensorInfo> e : outputs.entrySet()) {
            if (e.getValue().getShape().length == 3) {
                return e.getKey();
            }
        }
        throw new DjembedException(directory + " has neither a " + POOLED_OUTPUT + " output nor a [batch, tokens, hidden] one: "
                + outputs.keySet());
    }

    static int width(TensorInfo info, ModelDirectory dir, String name) {
        long[] shape = info.getShape();
        long last = shape[shape.length - 1];
        if (last > 0) {
            return (int) last;
        }
        if (dir.hiddenSize() > 0) {
            return dir.hiddenSize();
        }
        throw new DjembedException("Output '" + name + "' has a dynamic width and config.json declares no hidden_size");
    }

    @Override
    public CompletableFuture<Embeddings> embedAsync(List<String> texts) {
        int count = texts.size();
        if (count == 0) {
            return CompletableFuture.completedFuture(new Embeddings(new float[0], 0, dimension, 0, -1));
        }
        String[] input = texts.toArray(new String[0]);
        for (int i = 0; i < count; i++) {
            if (input[i] == null) {
                throw new IllegalArgumentException("texts[" + i + "] is null");
            }
        }
        return scheduler.submitAsync(count, lifecycle, future -> tokenize(future, input));
    }

    private EmbedJob tokenize(CompletableFuture<Embeddings> future, String[] texts) {
        Encoded encoded = tokenizer.encode(texts, chunk);
        Sequences seqs = chunk
                ? Sequences.chunked(encoded.ids(), encoded.aux(), maxInputTokens, prefix.length, suffix.length)
                : Sequences.whole(encoded.ids());
        return new EmbedJob(future, encoded, seqs, texts.length, dimension, normalize);
    }

    /** A request in flight: its sequences and the vectors they accumulate into. */
    private static final class EmbedJob extends Job<Embeddings> {

        final long[][] ids;
        final Sequences seqs;
        final int count;
        final long tokens;
        final int firstTruncated;
        final float[] vectors;
        final float[] weights;

        EmbedJob(CompletableFuture<Embeddings> future, Encoded encoded, Sequences seqs, int count, int dimension, boolean normalize) {
            super(future, seqs.lengths(), seqs.size());
            this.ids = encoded.ids();
            this.seqs = seqs;
            this.count = count;
            this.tokens = encoded.tokens();
            this.firstTruncated = encoded.firstTruncated();
            this.vectors = new float[count * dimension];
            this.weights = normalize ? null : new float[count];
        }
    }

    /** Device side; runs on the scheduler thread only. */
    private final class Runner implements BatchRunner<Embeddings, EmbedJob> {

        private final float[] pooled = new float[dimension];

        @Override
        public void begin(int rows, int rowLength) {
            workspace.begin(rows, rowLength);
        }

        @Override
        public void putRow(int row, EmbedJob job, int s) {
            Sequences seqs = job.seqs;
            boolean wrap = seqs.wrap(s);
            workspace.putRow(row, job.ids[seqs.source(s)], seqs.from(s), seqs.to(s),
                    wrap ? prefix : null, wrap ? suffix : null, null);
        }

        @Override
        public void run() {
            workspace.run();
        }

        @Override
        public void readRow(int row, EmbedJob job, int s) {
            readVector(row, job.seqs.length(s));
            accumulate(job, job.seqs.source(s), job.seqs.weight(s));
        }

        @Override
        public Embeddings finish(EmbedJob job) {
            for (int i = 0; i < job.count; i++) {
                if (normalize) {
                    l2Normalize(job.vectors, i * dimension, dimension);
                } else {
                    scale(job.vectors, i * dimension, dimension, 1f / job.weights[i]);
                }
            }
            return new Embeddings(job.vectors, job.count, dimension, job.tokens, job.firstTruncated);
        }

        /** Loads row {@code row} of the last forward pass into {@link #pooled}, pooling token vectors when needed. */
        private void readVector(int row, int length) {
            if (pooling == null) {
                long at = workspace.rowOffset(row);
                for (int j = 0; j < dimension; j++) {
                    pooled[j] = workspace.output(at + j);
                }
                return;
            }
            switch (pooling) {
                case CLS -> copyToken(row, 0);
                case LAST_TOKEN -> copyToken(row, length - 1);
                case MEAN -> {
                    Arrays.fill(pooled, 0f);
                    for (int t = 0; t < length; t++) {
                        long at = workspace.tokenOffset(row, t);
                        for (int j = 0; j < dimension; j++) {
                            pooled[j] += workspace.output(at + j);
                        }
                    }
                    scale(pooled, 0, dimension, 1f / length);
                }
            }
        }

        private void copyToken(int row, int token) {
            long at = workspace.tokenOffset(row, token);
            for (int j = 0; j < dimension; j++) {
                pooled[j] = workspace.output(at + j);
            }
        }

        /**
         * Adds {@link #pooled} into the vector of input {@code owner} with {@code weight}. Normalised output averages
         * unit vectors (and is re-normalised at the end); raw output averages the vectors as they are.
         */
        private void accumulate(EmbedJob job, int owner, int weight) {
            if (normalize) {
                l2Normalize(pooled, 0, dimension);
            } else {
                job.weights[owner] += weight;
            }
            int base = owner * dimension;
            for (int j = 0; j < dimension; j++) {
                job.vectors[base + j] += weight * pooled[j];
            }
        }
    }

    private static void l2Normalize(float[] v, int from, int length) {
        double sum = 0;
        for (int j = from; j < from + length; j++) {
            sum += v[j] * v[j];
        }
        if (sum > 0) {
            scale(v, from, length, (float) (1.0 / Math.sqrt(sum)));
        }
    }

    private static void scale(float[] v, int from, int length, float factor) {
        for (int j = from; j < from + length; j++) {
            v[j] *= factor;
        }
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int maxInputTokens() {
        return maxInputTokens;
    }

    @Override
    public int queuedInputs() {
        return scheduler.queuedInputs();
    }

    /** Fails requests not yet completed, then releases the device and the tokenizer. */
    @Override
    public void close() {
        if (!lifecycle.close()) {
            return;
        }
        scheduler.close();
        workspace.close();
        model.close();
        tokenizer.close();
    }
}

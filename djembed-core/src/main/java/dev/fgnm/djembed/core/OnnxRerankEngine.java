package dev.fgnm.djembed.core;

import ai.onnxruntime.TensorInfo;
import dev.fgnm.djembed.core.internal.BatchRunner;
import dev.fgnm.djembed.core.internal.Encoded;
import dev.fgnm.djembed.core.internal.HfTokenizer;
import dev.fgnm.djembed.core.internal.Job;
import dev.fgnm.djembed.core.internal.Lifecycle;
import dev.fgnm.djembed.core.internal.Limits;
import dev.fgnm.djembed.core.internal.ModelDirectory;
import dev.fgnm.djembed.core.internal.OnnxModel;
import dev.fgnm.djembed.core.internal.Scheduler;
import dev.fgnm.djembed.core.internal.Sequences;
import dev.fgnm.djembed.core.internal.Workspace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * {@link RerankEngine} running an ONNX cross-encoder (sequence classification with a single logit) in-process.
 *
 * <p>Requests are tokenized on a shared CPU pool, then batched across callers by a {@link Scheduler} that owns the
 * device: while one round runs on the device, the next requests are being tokenized.
 */
public final class OnnxRerankEngine implements RerankEngine {

    private static final Logger log = LoggerFactory.getLogger(OnnxRerankEngine.class);

    private static final String LOGITS_OUTPUT = "logits";

    private final HfTokenizer tokenizer;
    private final OnnxModel model;
    private final Workspace workspace;
    private final Scheduler<RerankScores, RerankJob> scheduler;
    private final Lifecycle lifecycle = new Lifecycle();

    private final int maxInputTokens;
    private final boolean typeIds;
    private final ScoreActivation activation;

    private OnnxRerankEngine(String name, HfTokenizer tokenizer, OnnxModel model, Workspace workspace, int maxInputTokens,
                             RerankOptions options, EngineObserver observer) {
        this.tokenizer = tokenizer;
        this.model = model;
        this.workspace = workspace;
        this.maxInputTokens = maxInputTokens;
        this.typeIds = model.takesTokenTypeIds();
        this.activation = options.activation();
        EngineOptions engine = options.engine();
        this.scheduler = new Scheduler<>(name, new Runner(), engine.maxBatchSize(), engine.tokenBudget(),
                engine.maxQueuedInputs(), CpuPool.executor(), observer);
    }

    /**
     * Loads the model folder at {@code directory} (see the README for the expected layout).
     */
    public static OnnxRerankEngine load(Path directory, RerankOptions options) {
        return load(directory, options, EngineObserver.NONE);
    }

    /**
     * Loads the model folder at {@code directory}, reporting its activity to {@code observer}.
     */
    public static OnnxRerankEngine load(Path directory, RerankOptions options, EngineObserver observer) {
        ModelDirectory dir = ModelDirectory.open(directory);
        EngineOptions engine = options.engine();
        int maxInputTokens = Limits.maxInputTokens(dir, engine);

        HfTokenizer tokenizer = HfTokenizer.load(dir.tokenizerFile(), maxInputTokens, true);
        OnnxModel model = null;
        Workspace workspace = null;
        try {
            model = OnnxModel.load(dir.onnxFile(), engine.device());
            String outputName = logitsOutput(model.outputs(), directory);
            workspace = new Workspace(model, outputName, 1,
                    Limits.tokenCapacity(engine, maxInputTokens), maxInputTokens, engine.maxBatchSize(), dir.padTokenId());
            String name = directory.getFileName().toString();
            log.info("Rerank engine {}: maxInputTokens={} output={} activation={}",
                    name, maxInputTokens, outputName, options.activation());
            return new OnnxRerankEngine(name, tokenizer, model, workspace, maxInputTokens, options, observer);
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

    private static String logitsOutput(Map<String, TensorInfo> outputs, Path directory) {
        String name = outputs.containsKey(LOGITS_OUTPUT) ? LOGITS_OUTPUT
                : outputs.entrySet().stream()
                        .filter(e -> e.getValue().getShape().length == 2)
                        .map(Map.Entry::getKey)
                        .findFirst()
                        .orElseThrow(() -> new DjembedException(directory + " has no [batch, labels] output: " + outputs.keySet()));
        long[] shape = outputs.get(name).getShape();
        if (shape.length != 2 || shape[1] != 1) {
            throw new DjembedException(directory + ": output '" + name + "' must be [batch, 1] (a single relevance logit), found "
                    + Arrays.toString(shape));
        }
        return name;
    }

    @Override
    public CompletableFuture<RerankScores> scoreAsync(String query, List<String> documents) {
        if (query == null) {
            throw new IllegalArgumentException("query is null");
        }
        int count = documents.size();
        if (count == 0) {
            return CompletableFuture.completedFuture(new RerankScores(new float[0], 0));
        }
        String[] input = documents.toArray(new String[0]);
        for (int i = 0; i < count; i++) {
            if (input[i] == null) {
                throw new IllegalArgumentException("documents[" + i + "] is null");
            }
        }
        return scheduler.submitAsync(count, lifecycle, future -> tokenize(future, query, input));
    }

    private RerankJob tokenize(CompletableFuture<RerankScores> future, String query, String[] documents) {
        Encoded encoded = tokenizer.encodePairs(query, documents, typeIds);
        return new RerankJob(future, encoded, Sequences.whole(encoded.ids()));
    }

    /** A request in flight: its (query, document) pairs and their scores. */
    private static final class RerankJob extends Job<RerankScores> {

        final long[][] ids;
        final long[][] types;
        final long tokens;
        final float[] scores;

        RerankJob(CompletableFuture<RerankScores> future, Encoded encoded, Sequences seqs) {
            super(future, seqs.lengths(), seqs.size());
            this.ids = encoded.ids();
            this.types = encoded.aux();
            this.tokens = encoded.tokens();
            this.scores = new float[encoded.ids().length];
        }
    }

    /** Device side; runs on the scheduler thread only. */
    private final class Runner implements BatchRunner<RerankScores, RerankJob> {

        @Override
        public void begin(int rows, int rowLength) {
            workspace.begin(rows, rowLength);
        }

        @Override
        public void putRow(int row, RerankJob job, int s) {
            long[] ids = job.ids[s];
            workspace.putRow(row, ids, 0, ids.length, null, null, job.types != null ? job.types[s] : null);
        }

        @Override
        public void run() {
            workspace.run();
        }

        @Override
        public void readRow(int row, RerankJob job, int s) {
            job.scores[s] = activate(workspace.output(workspace.rowOffset(row)));
        }

        @Override
        public RerankScores finish(RerankJob job) {
            return new RerankScores(job.scores, job.tokens);
        }
    }

    private float activate(float logit) {
        return switch (activation) {
            case SIGMOID -> (float) (1.0 / (1.0 + Math.exp(-logit)));
            case NONE -> logit;
        };
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

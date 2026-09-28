package dev.fgnm.djembed.server.config;

import dev.fgnm.djembed.core.Device;
import dev.fgnm.djembed.core.EmbeddingOptions;
import dev.fgnm.djembed.core.EngineOptions;
import dev.fgnm.djembed.core.LongInputStrategy;
import dev.fgnm.djembed.core.PoolingMode;
import dev.fgnm.djembed.core.RerankOptions;
import dev.fgnm.djembed.core.ScoreActivation;

import java.nio.file.Path;
import java.util.Objects;

/**
 * A model served under {@code name}, which is what clients pass in the {@code model} request field.
 * {@code path} points to the model directory; a relative path is resolved against the directory of the
 * configuration file. Unset settings fall back to the engine defaults.
 */
public record ModelConfig(
        String name,
        ModelTask task,
        Path path,
        String device,
        Integer maxBatchSize,
        Integer tokenBudget,
        Integer maxInputTokens,
        Integer maxQueuedInputs,
        LongInputStrategy longInput,
        PoolingMode pooling,
        Boolean normalize,
        ScoreActivation activation) {

    public ModelConfig {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("models[].name is required");
        }
        Objects.requireNonNull(task, () -> "models[" + name + "].task is required");
        Objects.requireNonNull(path, () -> "models[" + name + "].path is required");
        if (device != null) {
            Device.parse(device);
        }
        // A compact constructor sees its parameters, not yet the fields: everything below works on the former.
        if (task == ModelTask.RERANK) {
            rejectFor(name, "long-input", longInput, "rerank");
            rejectFor(name, "pooling", pooling, "rerank");
            rejectFor(name, "normalize", normalize, "rerank");
        } else {
            rejectFor(name, "activation", activation, "embed");
        }
        try {
            engineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("models[" + name + "]: " + e.getMessage(), e);
        }
    }

    private static void rejectFor(String name, String key, Object value, String taskName) {
        if (value != null) {
            throw new IllegalArgumentException("models[" + name + "]." + key + " does not apply to " + taskName + " models");
        }
    }

    EngineOptions engineOptions() {
        return engineOptions(device, maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs);
    }

    private static EngineOptions engineOptions(String device, Integer maxBatchSize, Integer tokenBudget,
                                               Integer maxInputTokens, Integer maxQueuedInputs) {
        EngineOptions options = EngineOptions.defaults();
        if (device != null) {
            options = options.withDevice(Device.parse(device));
        }
        if (maxBatchSize != null) {
            options = options.withMaxBatchSize(maxBatchSize);
        }
        if (tokenBudget != null) {
            options = options.withTokenBudget(tokenBudget);
        }
        if (maxInputTokens != null) {
            options = options.withMaxInputTokens(maxInputTokens);
        }
        if (maxQueuedInputs != null) {
            options = options.withMaxQueuedInputs(maxQueuedInputs);
        }
        return options;
    }

    public EmbeddingOptions embeddingOptions() {
        EmbeddingOptions options = EmbeddingOptions.defaults().withEngine(engineOptions());
        if (longInput != null) {
            options = options.withLongInput(longInput);
        }
        if (pooling != null) {
            options = options.withPooling(pooling);
        }
        if (normalize != null) {
            options = options.withNormalize(normalize);
        }
        return options;
    }

    public RerankOptions rerankOptions() {
        RerankOptions options = RerankOptions.defaults().withEngine(engineOptions());
        if (activation != null) {
            options = options.withActivation(activation);
        }
        return options;
    }

    ModelConfig resolveAgainst(Path baseDir) {
        return path.isAbsolute() ? this : new ModelConfig(name, task, baseDir.resolve(path).normalize(), device,
                maxBatchSize, tokenBudget, maxInputTokens, maxQueuedInputs, longInput, pooling, normalize, activation);
    }
}

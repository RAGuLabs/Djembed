package com.ragulabs.djembed.core.internal;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ragulabs.djembed.core.DjembedException;
import com.ragulabs.djembed.core.PoolingMode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A model folder laid out like a Hugging Face / sentence-transformers export: the ONNX graph
 * ({@code model.onnx} or {@code onnx/model.onnx}), {@code tokenizer.json}, and the JSON files
 * the metadata below is read from. Nothing here is configured by hand.
 */
public final class ModelDirectory {

    private static final ObjectMapper JSON = new ObjectMapper();

    // Position ids of RoBERTa-family models start after the padding index, so
    // max_position_embeddings overstates the usable length by pad_token_id + 1.
    private static final Set<String> OFFSET_POSITION_MODELS = Set.of("roberta", "xlm-roberta", "camembert");

    // tokenizer_config.json uses a huge sentinel (1e30) when no limit is declared.
    private static final long UNSET_MODEL_MAX_LENGTH = 1_000_000L;

    private final Path root;
    private final Path onnxFile;
    private final Path tokenizerFile;
    private final int padTokenId;
    private final int hiddenSize;
    private final int maxInputTokens;
    private final PoolingMode pooling;

    private ModelDirectory(Path root, Path onnxFile, Path tokenizerFile, int padTokenId, int hiddenSize,
                           int maxInputTokens, PoolingMode pooling) {
        this.root = root;
        this.onnxFile = onnxFile;
        this.tokenizerFile = tokenizerFile;
        this.padTokenId = padTokenId;
        this.hiddenSize = hiddenSize;
        this.maxInputTokens = maxInputTokens;
        this.pooling = pooling;
    }

    public static ModelDirectory open(Path root) {
        if (!Files.isDirectory(root)) {
            throw new DjembedException("Model directory not found: " + root);
        }
        Path onnx = firstExisting(root, "model.onnx", "onnx/model.onnx");
        Path tokenizer = firstExisting(root, "tokenizer.json");

        JsonNode config = readJson(root.resolve("config.json"));
        JsonNode tokenizerConfig = readJson(root.resolve("tokenizer_config.json"));
        JsonNode sentenceBertConfig = readJson(root.resolve("sentence_bert_config.json"));
        JsonNode poolingConfig = readJson(root.resolve("1_Pooling/config.json"));

        int padTokenId = config.path("pad_token_id").asInt(0);
        int hiddenSize = config.path("hidden_size").asInt(0);
        int maxInputTokens = resolveMaxInputTokens(root, config, tokenizerConfig, sentenceBertConfig);

        return new ModelDirectory(root, onnx, tokenizer, padTokenId, hiddenSize, maxInputTokens, resolvePooling(root, poolingConfig));
    }

    static int resolveMaxInputTokens(Path root, JsonNode config, JsonNode tokenizerConfig, JsonNode sentenceBertConfig) {
        if (sentenceBertConfig.has("max_seq_length")) {
            return sentenceBertConfig.get("max_seq_length").asInt();
        }
        long modelMaxLength = tokenizerConfig.path("model_max_length").asLong(0);
        if (modelMaxLength > 0 && modelMaxLength < UNSET_MODEL_MAX_LENGTH) {
            return (int) modelMaxLength;
        }
        int positions = config.path("max_position_embeddings").asInt(0);
        if (positions > 0) {
            if (OFFSET_POSITION_MODELS.contains(config.path("model_type").asText())) {
                return positions - config.path("pad_token_id").asInt(1) - 1;
            }
            return positions;
        }
        throw new DjembedException("Cannot determine the token limit of " + root
                + ": none of sentence_bert_config.json:max_seq_length, tokenizer_config.json:model_max_length,"
                + " config.json:max_position_embeddings is present");
    }

    static PoolingMode resolvePooling(Path root, JsonNode poolingConfig) {
        if (poolingConfig.isMissingNode()) {
            return null;
        }
        List<PoolingMode> enabled = new ArrayList<>(1);
        if (poolingConfig.path("pooling_mode_cls_token").asBoolean(false)) {
            enabled.add(PoolingMode.CLS);
        }
        if (poolingConfig.path("pooling_mode_mean_tokens").asBoolean(false)) {
            enabled.add(PoolingMode.MEAN);
        }
        if (poolingConfig.path("pooling_mode_lasttoken").asBoolean(false)) {
            enabled.add(PoolingMode.LAST_TOKEN);
        }
        if (enabled.size() != 1) {
            throw new DjembedException("Unsupported pooling in " + root.resolve("1_Pooling/config.json")
                    + ": exactly one of cls_token, mean_tokens, lasttoken must be enabled, found " + enabled);
        }
        return enabled.getFirst();
    }

    private static Path firstExisting(Path root, String... candidates) {
        for (String candidate : candidates) {
            Path p = root.resolve(candidate);
            if (Files.isRegularFile(p)) {
                return p;
            }
        }
        throw new DjembedException("None of " + List.of(candidates) + " found in " + root);
    }

    private static JsonNode readJson(Path file) {
        if (!Files.isRegularFile(file)) {
            return JSON.missingNode();
        }
        try {
            return JSON.readTree(file.toFile());
        } catch (IOException e) {
            throw new DjembedException("Cannot parse " + file + ": " + e.getMessage(), e);
        }
    }

    public Path root() {
        return root;
    }

    public Path onnxFile() {
        return onnxFile;
    }

    public Path tokenizerFile() {
        return tokenizerFile;
    }

    public int padTokenId() {
        return padTokenId;
    }

    /** {@code config.json:hidden_size}, or 0 when absent. */
    public int hiddenSize() {
        return hiddenSize;
    }

    public int maxInputTokens() {
        return maxInputTokens;
    }

    /** Pooling declared by {@code 1_Pooling/config.json}, or {@code null} when the folder has none. */
    public PoolingMode pooling() {
        return pooling;
    }
}

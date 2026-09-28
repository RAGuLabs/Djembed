package com.ragulabs.djembed.core.internal;

import com.ragulabs.djembed.core.DjembedException;
import com.ragulabs.djembed.core.PoolingMode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ModelDirectoryTest {

    @TempDir
    Path dir;

    @Test
    void readsTheLimitFromTokenizerConfig() throws IOException {
        layout("onnx/model.onnx");
        write("config.json", """
                {"model_type": "xlm-roberta", "max_position_embeddings": 8194, "pad_token_id": 1, "hidden_size": 1024}""");
        write("tokenizer_config.json", """
                {"model_max_length": 8192}""");

        ModelDirectory model = ModelDirectory.open(dir);

        assertEquals(dir.resolve("onnx/model.onnx"), model.onnxFile());
        assertEquals(8192, model.maxInputTokens());
        assertEquals(1, model.padTokenId());
        assertEquals(1024, model.hiddenSize());
        assertNull(model.pooling());
    }

    @Test
    void sentenceBertConfigWins() throws IOException {
        layout("model.onnx");
        write("sentence_bert_config.json", """
                {"max_seq_length": 256}""");
        write("tokenizer_config.json", """
                {"model_max_length": 512}""");

        assertEquals(256, ModelDirectory.open(dir).maxInputTokens());
    }

    @Test
    void robertaPositionsAreOffsetByThePaddingIndex() throws IOException {
        layout("model.onnx");
        write("config.json", """
                {"model_type": "xlm-roberta", "max_position_embeddings": 8194, "pad_token_id": 1}""");
        write("tokenizer_config.json", """
                {"model_max_length": 1000000000000000019884624838656}""");

        assertEquals(8192, ModelDirectory.open(dir).maxInputTokens());
    }

    @Test
    void bertPositionsAreTakenAsIs() throws IOException {
        layout("model.onnx");
        write("config.json", """
                {"model_type": "bert", "max_position_embeddings": 512}""");

        assertEquals(512, ModelDirectory.open(dir).maxInputTokens());
    }

    @Test
    void readsPooling() throws IOException {
        layout("model.onnx");
        write("config.json", """
                {"max_position_embeddings": 512}""");
        write("1_Pooling/config.json", """
                {"word_embedding_dimension": 384, "pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}""");

        assertEquals(PoolingMode.MEAN, ModelDirectory.open(dir).pooling());
    }

    @Test
    void rejectsCombinedPooling() throws IOException {
        layout("model.onnx");
        write("config.json", """
                {"max_position_embeddings": 512}""");
        write("1_Pooling/config.json", """
                {"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true}""");

        assertThrows(DjembedException.class, () -> ModelDirectory.open(dir));
    }

    @Test
    void rejectsFoldersWithoutAModel() throws IOException {
        write("tokenizer.json", "{}");

        assertThrows(DjembedException.class, () -> ModelDirectory.open(dir));
    }

    @Test
    void rejectsFoldersWithoutALimit() throws IOException {
        layout("model.onnx");

        assertThrows(DjembedException.class, () -> ModelDirectory.open(dir));
    }

    private void layout(String onnxPath) throws IOException {
        write(onnxPath, "");
        write("tokenizer.json", "{}");
    }

    private void write(String relative, String content) throws IOException {
        Path file = dir.resolve(relative);
        Files.createDirectories(file.getParent());
        Files.writeString(file, content);
    }
}

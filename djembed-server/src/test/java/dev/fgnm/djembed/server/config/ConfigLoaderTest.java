package dev.fgnm.djembed.server.config;

import dev.fgnm.djembed.core.Device;
import dev.fgnm.djembed.core.EmbeddingOptions;
import dev.fgnm.djembed.core.LongInputStrategy;
import dev.fgnm.djembed.core.PoolingMode;
import dev.fgnm.djembed.core.RerankOptions;
import dev.fgnm.djembed.core.ScoreActivation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfigLoaderTest {

    @TempDir
    Path dir;

    @Test
    void loadsModelsAndResolvesRelativePaths() throws IOException {
        Path file = write("""
                server:
                  port: 9090
                models:
                  - name: bge-m3
                    task: embed
                    path: models/bge-m3
                  - name: reranker
                    task: RERANK
                    path: /opt/models/reranker
                """);

        DjembedConfig config = ConfigLoader.load(file);

        assertEquals("0.0.0.0", config.server().host());
        assertEquals(9090, config.server().port());
        assertEquals(60_000, config.server().requestTimeoutMs());
        assertTrue(config.server().apiKeys().isEmpty());
        assertEquals(2, config.models().size());

        ModelConfig embed = config.models().get(0);
        assertEquals(ModelTask.EMBED, embed.task());
        assertEquals(dir.toAbsolutePath().resolve("models/bge-m3"), embed.path());

        ModelConfig rerank = config.models().get(1);
        assertEquals(ModelTask.RERANK, rerank.task());
        assertEquals(Path.of("/opt/models/reranker"), rerank.path());
    }

    @Test
    void mapsEngineSettings() throws IOException {
        Path file = write("""
                models:
                  - name: e
                    task: embed
                    path: e
                    device: cuda:1
                    max-batch-size: 32
                    token-budget: 8192
                    max-input-tokens: 512
                    max-queued-inputs: 1000
                    long-input: chunk
                    pooling: mean
                    normalize: false
                  - name: r
                    task: rerank
                    path: r
                    activation: none
                """);

        DjembedConfig config = ConfigLoader.load(file);

        EmbeddingOptions embed = config.models().get(0).embeddingOptions();
        assertEquals(Device.cuda(1), embed.engine().device());
        assertEquals(32, embed.engine().maxBatchSize());
        assertEquals(8192, embed.engine().tokenBudget());
        assertEquals(512, embed.engine().maxInputTokens());
        assertEquals(1000, embed.engine().maxQueuedInputs());
        assertEquals(LongInputStrategy.CHUNK, embed.longInput());
        assertEquals(PoolingMode.MEAN, embed.pooling());
        assertFalse(embed.normalize());

        RerankOptions rerank = config.models().get(1).rerankOptions();
        assertEquals(Device.cpu(), rerank.engine().device());
        assertEquals(ScoreActivation.NONE, rerank.activation());
    }

    @Test
    void rejectsSettingsOfTheOtherTask() throws IOException {
        Path file = write("""
                models:
                  - { name: r, task: rerank, path: r, pooling: cls }
                """);

        ConfigException e = assertThrows(ConfigException.class, () -> ConfigLoader.load(file));
        assertTrue(e.getMessage().contains("models[r].pooling does not apply to rerank models"), e.getMessage());
    }

    @Test
    void rejectsInvalidEngineSettings() throws IOException {
        Path file = write("""
                models:
                  - { name: e, task: embed, path: e, max-batch-size: 0 }
                """);

        ConfigException e = assertThrows(ConfigException.class, () -> ConfigLoader.load(file));
        assertTrue(e.getMessage().contains("models[e]: maxBatchSize must be >= 1"), e.getMessage());
    }

    @Test
    void readsApiKeysAndRejectsBlankOnes() throws IOException {
        DjembedConfig config = ConfigLoader.load(write("""
                server:
                  api-keys: [a, b]
                """));
        assertEquals(List.of("a", "b"), config.server().apiKeys());

        assertThrows(ConfigException.class, () -> ConfigLoader.load(write("""
                server:
                  api-keys: [" "]
                """)));
    }

    @Test
    void rejectsUnknownDevices() throws IOException {
        Path file = write("""
                models:
                  - { name: e, task: embed, path: e, device: tpu }
                """);

        assertThrows(ConfigException.class, () -> ConfigLoader.load(file));
    }

    @Test
    void emptyFileYieldsDefaults() throws IOException {
        DjembedConfig config = ConfigLoader.load(write(""));

        assertEquals(8080, config.server().port());
        assertTrue(config.models().isEmpty());
    }

    @Test
    void rejectsUnknownProperties() throws IOException {
        Path file = write("""
                server:
                  prot: 9090
                """);

        assertThrows(ConfigException.class, () -> ConfigLoader.load(file));
    }

    @Test
    void rejectsDuplicateModelNames() throws IOException {
        Path file = write("""
                models:
                  - { name: a, task: embed, path: x }
                  - { name: a, task: rerank, path: y }
                """);

        ConfigException e = assertThrows(ConfigException.class, () -> ConfigLoader.load(file));
        assertTrue(e.getMessage().contains("Duplicate model name: a"), e.getMessage());
    }

    @Test
    void rejectsModelWithoutTask() throws IOException {
        Path file = write("""
                models:
                  - { name: a, path: x }
                """);

        ConfigException e = assertThrows(ConfigException.class, () -> ConfigLoader.load(file));
        assertTrue(e.getMessage().contains("models[a].task is required"), e.getMessage());
    }

    private Path write(String yaml) throws IOException {
        return Files.writeString(dir.resolve("djembed.yaml"), yaml);
    }
}

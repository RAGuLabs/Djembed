package dev.fgnm.djembed.core;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("onnx")
class OnnxEmbeddingEngineTest {

    private static final String MODEL = "bge-m3";

    private static OnnxEmbeddingEngine engine;

    @BeforeAll
    static void load() {
        engine = OnnxEmbeddingEngine.load(TestModels.model(MODEL), EmbeddingOptions.defaults().withEngine(TestModels.engine()));
    }

    @AfterAll
    static void close() {
        if (engine != null) {
            engine.close();
        }
    }

    @Test
    void returnsUnitVectorsInInputOrder() {
        Embeddings e = engine.embed(List.of("Il gatto dorme sul divano.", "The cat sleeps on the sofa.", "Quarterly revenue grew 4%."));

        assertEquals(3, e.count());
        assertEquals(1024, e.dimension());
        for (int i = 0; i < e.count(); i++) {
            assertEquals(1.0, norm(e.vector(i)), 1e-4);
        }
        // Cross-lingual paraphrases sit closer than an unrelated sentence.
        assertTrue(cosine(e.vector(0), e.vector(1)) > cosine(e.vector(0), e.vector(2)));
        assertTrue(e.promptTokens() > 0);
    }

    @Test
    void paddingDoesNotChangeTheVector() {
        String shortText = "Una frase breve.";
        String longText = "Una frase molto più lunga, che obbliga le righe più corte dello stesso batch a essere "
                + "riempite con token di padding fino alla sua lunghezza, per verificare che la maschera li escluda.";

        float[] alone = engine.embed(List.of(shortText)).vector(0);
        float[] batched = engine.embed(List.of(longText, shortText)).vector(1);

        assertEquals(1.0, cosine(alone, batched), 1e-4);
    }

    @Test
    void reportsTruncatedInputs() {
        try (OnnxEmbeddingEngine small = OnnxEmbeddingEngine.load(TestModels.model(MODEL), EmbeddingOptions.defaults()
                .withEngine(TestModels.engine().withMaxInputTokens(16)))) {
            Embeddings fits = small.embed(List.of("Breve."));
            Embeddings cut = small.embed(List.of("Breve.", "Questa frase è decisamente più lunga di sedici token una volta tokenizzata."));

            assertEquals(-1, fits.firstTruncated());
            assertEquals(1, cut.firstTruncated());
            assertTrue(cut.promptTokens() <= 2 * 16);
        }
    }

    @Test
    void concurrentRequestsGetTheirOwnVectors() {
        List<List<String>> requests = new ArrayList<>();
        for (int r = 0; r < 24; r++) {
            List<String> texts = new ArrayList<>();
            for (int t = 0; t <= r % 5; t++) {
                texts.add("Richiesta " + r + ", testo " + t + ": " + "parola ".repeat(1 + (r * 7 + t * 13) % 40));
            }
            requests.add(texts);
        }
        List<Embeddings> sequential = requests.stream().map(engine::embed).toList();

        List<CompletableFuture<Embeddings>> concurrent = requests.stream().map(engine::embedAsync).toList();

        for (int r = 0; r < requests.size(); r++) {
            Embeddings expected = sequential.get(r);
            Embeddings actual = concurrent.get(r).join();
            assertEquals(expected.count(), actual.count());
            for (int i = 0; i < actual.count(); i++) {
                assertEquals(1.0, cosine(expected.vector(i), actual.vector(i)), 1e-4, "request " + r + " text " + i);
            }
        }
    }

    @Test
    void emptyBatch() {
        assertEquals(0, engine.embed(List.of()).count());
    }

    @Test
    void chunksInputsOverTheLimit() {
        try (OnnxEmbeddingEngine chunking = OnnxEmbeddingEngine.load(TestModels.model(MODEL), EmbeddingOptions.defaults()
                .withEngine(TestModels.engine().withMaxInputTokens(32))
                .withLongInput(LongInputStrategy.CHUNK))) {
            List<String> sentences = new ArrayList<>();
            for (int i = 0; i < 20; i++) {
                sentences.add("La frase numero " + i + " parla di diritto civile e contratti.");
            }
            String longText = String.join(" ", sentences);

            Embeddings e = chunking.embed(List.of(longText, sentences.getFirst()));

            assertEquals(1.0, norm(e.vector(0)), 1e-4);
            assertTrue(e.promptTokens() > 32, "the whole text is tokenized, not a truncated prefix");
            assertTrue(cosine(e.vector(0), e.vector(1)) > 0.5);
        }
    }

    static double norm(float[] v) {
        double s = 0;
        for (float x : v) {
            s += x * x;
        }
        return Math.sqrt(s);
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot / (norm(a) * norm(b));
    }
}

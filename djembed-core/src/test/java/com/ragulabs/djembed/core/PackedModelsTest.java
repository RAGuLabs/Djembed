package com.ragulabs.djembed.core;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static com.ragulabs.djembed.core.OnnxEmbeddingEngineTest.cosine;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Models converted to ONNX Runtime's packing mode must give what their padded fp16 source gives. Skipped unless the
 * {@code -fp16} and {@code -fp16-packed} folders exist under {@code -Pmodels}.
 */
@Tag("onnx")
class PackedModelsTest {

    @Test
    void packedEmbeddingsMatchPadded() {
        List<String> texts = mixedLengths();
        EmbeddingOptions options = EmbeddingOptions.defaults().withEngine(TestModels.engine());
        Embeddings padded;
        try (OnnxEmbeddingEngine engine = OnnxEmbeddingEngine.load(TestModels.model("bge-m3-fp16"), options)) {
            padded = engine.embed(texts);
        }
        try (OnnxEmbeddingEngine engine = OnnxEmbeddingEngine.load(TestModels.model("bge-m3-fp16-packed"), options)) {
            Embeddings packed = engine.embed(texts);
            for (int i = 0; i < texts.size(); i++) {
                assertEquals(1.0, cosine(padded.vector(i), packed.vector(i)), 1e-3, "text " + i);
            }
            // Alone or among much longer companions, a packed row sees exactly the same tokens.
            float[] alone = engine.embed(List.of(texts.getFirst())).vector(0);
            assertEquals(1.0, cosine(alone, packed.vector(0)), 1e-4);
        }
    }

    @Test
    void packedScoresMatchPadded() {
        String query = "Qual è il termine di prescrizione ordinario?";
        List<String> documents = mixedLengths();
        RerankOptions options = RerankOptions.defaults().withEngine(TestModels.engine());
        RerankScores padded;
        try (OnnxRerankEngine engine = OnnxRerankEngine.load(TestModels.model("bge-reranker-v2-m3-fp16"), options)) {
            padded = engine.score(query, documents);
        }
        try (OnnxRerankEngine engine = OnnxRerankEngine.load(TestModels.model("bge-reranker-v2-m3-fp16-packed"), options)) {
            RerankScores packed = engine.score(query, documents);
            for (int i = 0; i < documents.size(); i++) {
                assertEquals(padded.score(i), packed.score(i), 5e-3, "document " + i);
            }
        }
    }

    private static List<String> mixedLengths() {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 24; i++) {
            texts.add("Frase " + i + ": " + "il diritto si estingue per prescrizione ".repeat(1 + (i * 7) % 40));
        }
        return texts;
    }
}

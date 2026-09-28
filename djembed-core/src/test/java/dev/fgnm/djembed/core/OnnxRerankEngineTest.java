package dev.fgnm.djembed.core;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("onnx")
class OnnxRerankEngineTest {

    private static final String MODEL = "bge-reranker-v2-m3";
    private static final String QUERY = "Qual è il termine di prescrizione ordinario?";
    private static final String RELEVANT = "Salvi i casi in cui la legge dispone diversamente, i diritti si estinguono "
            + "per prescrizione con il decorso di dieci anni.";
    private static final String IRRELEVANT = "La ricetta prevede farina, uova e zucchero.";
    private static final String LONG = "Il contratto è l'accordo di due o più parti per costituire, regolare o estinguere "
            + "tra loro un rapporto giuridico patrimoniale. ".repeat(12);

    private static OnnxRerankEngine engine;

    @BeforeAll
    static void load() {
        engine = OnnxRerankEngine.load(TestModels.model(MODEL), RerankOptions.defaults().withEngine(TestModels.engine()));
    }

    @AfterAll
    static void close() {
        if (engine != null) {
            engine.close();
        }
    }

    @Test
    void relevantDocumentScoresHigher() {
        RerankScores scores = engine.score(QUERY, List.of(IRRELEVANT, RELEVANT));

        assertEquals(2, scores.count());
        // Reference values from an independent implementation on the same checkpoint: ~0.169 and ~1.6e-5.
        assertTrue(scores.score(1) > 0.1f);
        assertTrue(scores.score(0) < 1e-3f);
    }

    @Test
    void concurrentRequestsGetTheirOwnScores() {
        List<List<String>> requests = List.of(
                List.of(RELEVANT, IRRELEVANT), List.of(LONG), List.of(IRRELEVANT, LONG, RELEVANT),
                List.of(RELEVANT), List.of(LONG, LONG, IRRELEVANT), List.of(IRRELEVANT));
        List<RerankScores> sequential = requests.stream().map(docs -> engine.score(QUERY, docs)).toList();

        List<CompletableFuture<RerankScores>> concurrent = requests.stream().map(docs -> engine.scoreAsync(QUERY, docs)).toList();

        for (int r = 0; r < requests.size(); r++) {
            for (int i = 0; i < requests.get(r).size(); i++) {
                assertEquals(sequential.get(r).score(i), concurrent.get(r).join().score(i), 1e-3, "request " + r + " doc " + i);
            }
        }
    }

    @Test
    void scoresDoNotDependOnBatchCompanionsOrOrder() {
        float alone = engine.score(QUERY, List.of(RELEVANT)).score(0);
        RerankScores mixed = engine.score(QUERY, List.of(LONG, IRRELEVANT, RELEVANT, LONG));

        // CPU is bit-exact; on CUDA the kernels picked for a different batch shape move scores by ~1e-4.
        assertEquals(alone, mixed.score(2), 1e-3);
        assertEquals(mixed.score(0), mixed.score(3), 1e-6);
    }
}

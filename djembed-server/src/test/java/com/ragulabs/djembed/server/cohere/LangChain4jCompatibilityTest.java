package com.ragulabs.djembed.server.cohere;

import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.testing.junit5.server.ServerExtension;
import com.ragulabs.djembed.server.FakeEngines;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.cohere.CohereEmbeddingModel;
import dev.langchain4j.model.cohere.CohereScoringModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Drives the server with LangChain4j's Cohere client, the one RAGu uses, configured the way RAGu configures it.
 */
class LangChain4jCompatibilityTest {

    @RegisterExtension
    static final ServerExtension server = new ServerExtension() {
        @Override
        protected void configure(ServerBuilder sb) {
            sb.annotatedService(new CohereService(FakeEngines.registry()));
        }
    };

    @Test
    void embeddingModel() {
        CohereEmbeddingModel model = CohereEmbeddingModel.builder()
                .baseUrl(server.httpUri() + "/v1/")
                .apiKey("unused")
                .modelName(FakeEngines.EMBED)
                .build();

        List<Embedding> embeddings = model.embedAll(List.of(TextSegment.from("ab"), TextSegment.from("abcd"))).content();

        assertEquals(2, embeddings.size());
        assertArrayEquals(new float[]{2f, 1f, 0.5f}, embeddings.get(0).vector());
        assertArrayEquals(new float[]{4f, 1f, 0.5f}, embeddings.get(1).vector());
    }

    @Test
    void scoringModel() {
        CohereScoringModel model = CohereScoringModel.builder()
                .baseUrl(server.httpUri() + "/v1/")
                .apiKey("unused")
                .modelName(FakeEngines.RERANK)
                .build();

        List<Double> scores = model.scoreAll(List.of(TextSegment.from("dogs"), TextSegment.from("a cat")), "cat").content();

        // Input order, even though the server ranks "a cat" first.
        assertEquals(0.004, scores.get(0), 1e-6);
        assertEquals(0.9, scores.get(1), 1e-6);
    }
}

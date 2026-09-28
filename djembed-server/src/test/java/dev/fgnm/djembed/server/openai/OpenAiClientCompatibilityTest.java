package dev.fgnm.djembed.server.openai;

import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.testing.junit5.server.ServerExtension;
import dev.fgnm.djembed.server.FakeEngines;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.openaiofficial.OpenAiOfficialEmbeddingModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Drives the server with the official OpenAI Java SDK through LangChain4j, configured the way RAGu configures it.
 */
class OpenAiClientCompatibilityTest {

    @RegisterExtension
    static final ServerExtension server = new ServerExtension() {
        @Override
        protected void configure(ServerBuilder sb) {
            sb.annotatedService(new OpenAiService(FakeEngines.registry()));
        }
    };

    @Test
    void embeddingModel() {
        OpenAiOfficialEmbeddingModel model = OpenAiOfficialEmbeddingModel.builder()
                .baseUrl(server.httpUri() + "/v1")
                .apiKey("unused")
                .modelName(FakeEngines.EMBED)
                .build();

        List<Embedding> embeddings = model.embedAll(List.of(TextSegment.from("ab"), TextSegment.from("abcd"))).content();

        assertArrayEquals(new float[]{2f, 1f, 0.5f}, embeddings.get(0).vector());
        assertArrayEquals(new float[]{4f, 1f, 0.5f}, embeddings.get(1).vector());
    }
}

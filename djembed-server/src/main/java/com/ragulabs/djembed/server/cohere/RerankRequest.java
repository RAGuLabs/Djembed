package com.ragulabs.djembed.server.cohere;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Body of {@code /v1/rerank} and {@code /v2/rerank}.
 */
record RerankRequest(
        String model,
        String query,
        List<Document> documents,
        Integer topN,
        Boolean returnDocuments,
        List<String> rankFields,
        Integer maxChunksPerDoc,
        Integer maxTokensPerDoc,
        Integer priority) {

    /** A document given as a plain string or, in v1, as an object whose {@code text} field is ranked. */
    record Document(@JsonProperty("text") String text) {

        @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
        Document {
        }

        @JsonCreator(mode = JsonCreator.Mode.DELEGATING)
        static Document of(String text) {
            return new Document(text);
        }
    }
}

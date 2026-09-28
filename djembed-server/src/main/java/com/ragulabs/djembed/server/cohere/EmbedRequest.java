package com.ragulabs.djembed.server.cohere;

import java.util.List;

/**
 * Body of {@code /v1/embed} and {@code /v2/embed}.
 */
record EmbedRequest(
        String model,
        List<String> texts,
        List<Input> inputs,
        List<String> images,
        String inputType,
        List<String> embeddingTypes,
        Integer outputDimension,
        Integer maxTokens,
        String truncate,
        Integer priority) {

    /** A v2 input: content parts that together make one embedding. */
    record Input(List<Content> content) {
    }

    record Content(String type, String text) {
    }
}

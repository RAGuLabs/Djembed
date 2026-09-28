package dev.fgnm.djembed.server.openai;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;

import java.io.IOException;
import java.io.Serial;
import java.util.ArrayList;
import java.util.List;

/**
 * Body of {@code /v1/embeddings}.
 *
 * @param input a string or an array of strings; token-id arrays are refused
 */
record EmbeddingsRequest(
        String model,
        @JsonDeserialize(using = EmbeddingsRequest.InputDeserializer.class) List<String> input,
        String encodingFormat,
        Integer dimensions,
        String user) {

    /** Streams {@code input} straight into strings, without an intermediate JSON tree. */
    static final class InputDeserializer extends StdDeserializer<List<String>> {

        @Serial
        private static final long serialVersionUID = 1L;

        InputDeserializer() {
            super(List.class);
        }

        @Override
        public List<String> deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
            if (p.currentToken() == JsonToken.VALUE_STRING) {
                List<String> single = new ArrayList<>(1);
                single.add(p.getText());
                return single;
            }
            if (p.currentToken() != JsonToken.START_ARRAY) {
                return ctxt.reportInputMismatch(this, "input must be a string or an array of strings");
            }
            List<String> texts = new ArrayList<>();
            JsonToken token;
            while ((token = p.nextToken()) != JsonToken.END_ARRAY) {
                switch (token) {
                    case VALUE_STRING -> texts.add(p.getText());
                    case VALUE_NULL -> texts.add(null);
                    case VALUE_NUMBER_INT, START_ARRAY ->
                            ctxt.reportInputMismatch(this, "token-id inputs are not supported, send text");
                    default -> ctxt.reportInputMismatch(this, "input must be a string or an array of strings");
                }
            }
            return texts;
        }
    }
}

package com.ragulabs.djembed.server.json;

import com.fasterxml.jackson.core.JsonGenerator;

import java.io.IOException;

/**
 * Writes floats as JSON numbers without a {@code String} per value.
 *
 * <p>{@link JsonGenerator#writeNumber(float)} formats through {@code Float.toString}, one short-lived string per
 * number: a single 1024-dimensional vector means a thousand of them. Here each value is appended to a reused
 * {@link StringBuilder} (the JDK's shortest round-trip algorithm, the same digits {@code Float.toString} yields),
 * copied into a reused {@code char[]} and handed to the generator as a raw value, which still places the commas.
 *
 * <p>One instance per response; not thread-safe.
 */
public final class FloatWriter {

    private final StringBuilder digits = new StringBuilder(24);
    private final char[] chars = new char[24];

    public void write(JsonGenerator json, float value) throws IOException {
        if (!Float.isFinite(value)) {
            throw new IllegalStateException("Cannot write " + value + " as a JSON number");
        }
        digits.setLength(0);
        digits.append(value);
        int length = digits.length();
        digits.getChars(0, length, chars, 0);
        json.writeRawValue(chars, 0, length);
    }
}

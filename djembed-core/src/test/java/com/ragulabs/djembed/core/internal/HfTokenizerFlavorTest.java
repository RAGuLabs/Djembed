package com.ragulabs.djembed.core.internal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class HfTokenizerFlavorTest {

    /** A minimal word-level tokenizer: enough for DJL to load its native library and encode. */
    private static final String TOKENIZER = """
            {"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
             "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
             "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2}, "unk_token": "[UNK]"}}""";

    @TempDir
    Path dir;

    @Test
    void usesTheCpuBuildBundledInTheJar() throws IOException {
        assumeTrue(System.getenv(HfTokenizer.FLAVOR_SETTING) == null, "an explicit RUST_FLAVOR wins");

        // Loading a tokenizer loads DJL's native library; with the CPU flavor it is copied from the jar, not downloaded.
        try (HfTokenizer tokenizer = HfTokenizer.load(Files.writeString(dir.resolve("tokenizer.json"), TOKENIZER), 16, true)) {
            assertEquals("cpu", System.getProperty(HfTokenizer.FLAVOR_SETTING));
            assertArrayEquals(new long[]{1, 2}, tokenizer.encode(new String[]{"hello world"}, false).ids()[0]);
        }
    }
}

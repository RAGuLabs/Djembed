package com.ragulabs.djembed.core.internal;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.jni.TokenizersLibrary;
import com.ragulabs.djembed.core.DjembedException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

/**
 * Hugging Face {@code tokenizers} (Rust) through DJL's JNI binding.
 *
 * <p>Calls go to {@link TokenizersLibrary} directly rather than through DJL's {@code Encoding}: that class eagerly
 * copies every field of every encoding into the heap (token strings, char spans, masks, overflow), while only the
 * ids, and sometimes the word or type ids, are needed here. Batches are encoded in parallel on the Rust side.
 */
public final class HfTokenizer implements AutoCloseable {

    private static final TokenizersLibrary LIB = TokenizersLibrary.LIB;

    private final HuggingFaceTokenizer tokenizer;
    private final long handle;
    private final boolean truncates;
    private final long[] prefix;
    private final long[] suffix;

    private HfTokenizer(HuggingFaceTokenizer tokenizer, boolean truncates) {
        this.tokenizer = tokenizer;
        this.handle = tokenizer.getHandle();
        this.truncates = truncates;
        long[][] specials = singleSequenceSpecials();
        this.prefix = specials[0];
        this.suffix = specials[1];
    }

    /**
     * @param maxLength token limit, special tokens included
     * @param truncate  cut longer inputs to {@code maxLength} (longest sequence first for pairs);
     *                  when {@code false} inputs are returned whole
     */
    public static HfTokenizer load(Path tokenizerJson, int maxLength, boolean truncate) {
        Map<String, String> options = Map.of(
                "addSpecialTokens", "true",
                "padding", "false",
                "truncation", truncate ? "true" : "false",
                "maxLength", Integer.toString(maxLength),
                "modelMaxLength", Integer.toString(maxLength));
        try {
            return new HfTokenizer(HuggingFaceTokenizer.newInstance(tokenizerJson, options), truncate);
        } catch (IOException e) {
            throw new DjembedException("Cannot load tokenizer " + tokenizerJson + ": " + e.getMessage(), e);
        }
    }

    /**
     * Special tokens the post-processor wraps around a single sequence ({@code [CLS] … [SEP]}, {@code <s> … </s>}),
     * read from the special-token mask of a probe encoding so any template works.
     */
    private long[][] singleSequenceSpecials() {
        long encoding = LIB.encode(handle, "a", true);
        try {
            long[] ids = LIB.getTokenIds(encoding);
            long[] special = LIB.getSpecialTokenMask(encoding);
            int lead = 0;
            while (lead < special.length && special[lead] == 1) {
                lead++;
            }
            int trail = 0;
            while (trail < special.length - lead && special[special.length - 1 - trail] == 1) {
                trail++;
            }
            return new long[][]{Arrays.copyOf(ids, lead), Arrays.copyOfRange(ids, ids.length - trail, ids.length)};
        } finally {
            LIB.deleteEncoding(encoding);
        }
    }

    /**
     * @param withWordIds also return word ids, for splitting long inputs at word boundaries
     */
    public Encoded encode(String[] texts, boolean withWordIds) {
        return collect(LIB.batchEncode(handle, texts, true), withWordIds ? Aux.WORD_IDS : Aux.NONE);
    }

    /**
     * Encodes {@code (query, document)} pairs.
     *
     * @param withTypeIds also return token type ids, for models that take {@code token_type_ids}
     */
    public Encoded encodePairs(String query, String[] documents, boolean withTypeIds) {
        String[] queries = new String[documents.length];
        Arrays.fill(queries, query);
        return collect(LIB.batchEncodePair(handle, queries, documents, true), withTypeIds ? Aux.TYPE_IDS : Aux.NONE);
    }

    private enum Aux { NONE, WORD_IDS, TYPE_IDS }

    private Encoded collect(long[] encodings, Aux aux) {
        long[][] ids = new long[encodings.length][];
        long[][] extra = aux == Aux.NONE ? null : new long[encodings.length][];
        long tokens = 0;
        int firstTruncated = -1;
        try {
            for (int i = 0; i < encodings.length; i++) {
                ids[i] = LIB.getTokenIds(encodings[i]);
                tokens += ids[i].length;
                // A truncated encoding keeps what it cut as overflow.
                if (truncates && firstTruncated < 0 && LIB.getOverflowCount(encodings[i]) > 0) {
                    firstTruncated = i;
                }
                if (aux == Aux.WORD_IDS) {
                    extra[i] = LIB.getWordIds(encodings[i]);
                } else if (aux == Aux.TYPE_IDS) {
                    extra[i] = LIB.getTypeIds(encodings[i]);
                }
            }
        } finally {
            for (long encoding : encodings) {
                LIB.deleteEncoding(encoding);
            }
        }
        return new Encoded(ids, extra, tokens, firstTruncated);
    }

    /** Special tokens before the text of a single-sequence encoding. */
    public long[] prefix() {
        return prefix.clone();
    }

    /** Special tokens after the text of a single-sequence encoding. */
    public long[] suffix() {
        return suffix.clone();
    }

    @Override
    public void close() {
        tokenizer.close();
    }
}

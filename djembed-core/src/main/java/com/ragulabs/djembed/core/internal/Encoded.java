package com.ragulabs.djembed.core.internal;

/**
 * Token ids of a tokenizer batch call, one row per input.
 *
 * @param ids            token ids, special tokens included
 * @param aux            word ids (for chunking) or token type ids (for pair models), per input; {@code null} when
 *                       not requested
 * @param tokens         total number of ids across all rows
 * @param firstTruncated index of the first input the tokenizer cut to its limit, or -1
 */
public record Encoded(long[][] ids, long[][] aux, long tokens, int firstTruncated) {
}

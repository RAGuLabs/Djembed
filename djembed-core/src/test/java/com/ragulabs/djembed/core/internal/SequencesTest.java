package com.ragulabs.djembed.core.internal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SequencesTest {

    @Test
    void shortInputsStayWhole() {
        long[][] ids = {{0, 5, 6, 2}};
        long[][] words = {{-1, 0, 1, -1}};

        Sequences seqs = Sequences.chunked(ids, words, 8, 1, 1);

        assertEquals(1, seqs.size());
        assertFalse(seqs.wrap(0));
        assertEquals(4, seqs.length(0));
    }

    @Test
    void longInputsSplitOnWordBoundaries() {
        // <s> w0 w0 w1 w1 w1 w2 w3 </s>: 7 text tokens, window of 4 text tokens (6 − 2 specials).
        long[][] ids = {{0, 10, 11, 12, 13, 14, 15, 16, 2}};
        long[][] words = {{-1, 0, 0, 1, 1, 1, 2, 3, -1}};

        Sequences seqs = Sequences.chunked(ids, words, 6, 1, 1);

        // A cut after 4 text tokens would split w1, so the first window stops before it;
        // the second takes w1 w1 w1 w2 and w3 is left for a third.
        assertEquals(3, seqs.size());
        assertRange(seqs, 0, 1, 3);
        assertRange(seqs, 1, 3, 7);
        assertRange(seqs, 2, 7, 8);
        for (int s = 0; s < seqs.size(); s++) {
            assertTrue(seqs.wrap(s));
            assertEquals(seqs.weight(s) + 2, seqs.length(s));
            assertTrue(seqs.length(s) <= 6);
        }
    }

    @Test
    void wordLongerThanTheWindowIsCutHard() {
        long[][] ids = {{0, 10, 11, 12, 13, 14, 2}};
        long[][] words = {{-1, 0, 0, 0, 0, 0, -1}};

        Sequences seqs = Sequences.chunked(ids, words, 4, 1, 1);

        assertEquals(3, seqs.size());
        assertRange(seqs, 0, 1, 3);
        assertRange(seqs, 1, 3, 5);
        assertRange(seqs, 2, 5, 6);
    }

    @Test
    void rejectsLimitsWithNoRoomForText() {
        assertThrows(IllegalArgumentException.class,
                () -> Sequences.chunked(new long[][]{{0, 2}}, new long[][]{{-1, -1}}, 2, 1, 1));
    }

    @Test
    void growsPastInitialCapacity() {
        long[][] ids = {new long[102]};
        long[][] words = {new long[102]};
        for (int i = 0; i < 102; i++) {
            words[0][i] = i;
        }

        Sequences seqs = Sequences.chunked(ids, words, 12, 1, 1);

        assertEquals(10, seqs.size());
        assertEquals(101, seqs.to(9));
    }

    private static void assertRange(Sequences seqs, int s, int from, int to) {
        assertEquals(from, seqs.from(s), "from of " + s);
        assertEquals(to, seqs.to(s), "to of " + s);
    }
}

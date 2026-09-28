package dev.fgnm.djembed.server.cohere;

import dev.fgnm.djembed.core.RerankScores;

import java.util.Arrays;

final class Ranking {

    private Ranking() {
    }

    /**
     * Document indices by descending score, ties by ascending index. Sorts packed primitive keys: the high half is
     * the score mapped to an int that orders like the float, inverted for descending order; the low half is the index.
     */
    static int[] descending(RerankScores scores) {
        int count = scores.count();
        long[] keys = new long[count];
        for (int i = 0; i < count; i++) {
            int bits = Float.floatToIntBits(scores.score(i));
            int ordered = bits ^ ((bits >> 31) & 0x7fffffff);
            keys[i] = ((long) ~ordered << 32) | i;
        }
        Arrays.sort(keys);
        int[] order = new int[count];
        for (int i = 0; i < count; i++) {
            order[i] = (int) keys[i];
        }
        return order;
    }
}

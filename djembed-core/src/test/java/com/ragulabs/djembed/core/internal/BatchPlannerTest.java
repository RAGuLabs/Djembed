package com.ragulabs.djembed.core.internal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchPlannerTest {

    private final BatchPlanner planner = new BatchPlanner();

    @Test
    void sortsByLengthAndKeepsTiesInInputOrder() {
        planner.plan(new int[]{5, 2, 9, 2}, 4, 10, 1_000);

        assertArrayEquals(new int[]{1, 3, 0, 2}, order(4));
        assertEquals(1, planner.batches());
    }

    @Test
    void packsUnderTheTokenBudget() {
        // Sorted lengths: 10, 10, 10, 40, 40. Budget 60: three rows of 10 fit (30), a fourth row
        // would pad everything to 40 (160). Then 2 × 40 = 80 > 60, so the forties go alone.
        int[] lengths = {40, 10, 10, 40, 10};
        planner.plan(lengths, 5, 64, 60);

        assertEquals(3, planner.batches());
        assertEquals(3, planner.rows(0));
        assertEquals(1, planner.rows(1));
        assertEquals(1, planner.rows(2));
        for (int b = 0; b < planner.batches(); b++) {
            int longest = lengths[planner.sequence(planner.start(b) + planner.rows(b) - 1)];
            assertTrue(planner.rows(b) == 1 || (long) planner.rows(b) * longest <= 60);
        }
    }

    @Test
    void respectsMaxRows() {
        planner.plan(new int[]{1, 1, 1, 1, 1}, 5, 2, 1_000);

        assertEquals(3, planner.batches());
        assertEquals(2, planner.rows(0));
        assertEquals(2, planner.rows(1));
        assertEquals(1, planner.rows(2));
    }

    @Test
    void oversizedSequenceGetsItsOwnBatch() {
        planner.plan(new int[]{500, 3}, 2, 8, 100);

        assertEquals(2, planner.batches());
        assertEquals(0, planner.sequence(1));
        assertEquals(1, planner.rows(1));
    }

    @Test
    void usesOnlyTheFirstCountLengths() {
        planner.plan(new int[]{4, 2, 99, 99}, 2, 8, 100);

        assertArrayEquals(new int[]{1, 0}, order(2));
        assertEquals(1, planner.batches());
    }

    @Test
    void reusesItsArraysAcrossCalls() {
        planner.plan(new int[]{3, 1, 2}, 3, 8, 100);
        planner.plan(new int[]{7, 6}, 2, 8, 100);

        assertArrayEquals(new int[]{1, 0}, order(2));
        assertEquals(1, planner.batches());
        assertEquals(2, planner.rows(0));
    }

    @Test
    void staticSortMatches() {
        assertArrayEquals(new int[]{1, 3, 0, 2}, BatchPlanner.sortedByLength(new int[]{5, 2, 9, 2}, 4));
    }

    private int[] order(int count) {
        int[] order = new int[count];
        for (int i = 0; i < count; i++) {
            order[i] = planner.sequence(i);
        }
        return order;
    }
}

package com.ragulabs.djembed.bench;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CorpusTest {

    @Test
    void sameSeedSameRequests() {
        assertEquals(Corpus.payloads(Workload.INGEST, 42, 5), Corpus.payloads(Workload.INGEST, 42, 5));
        assertNotEquals(Corpus.payloads(Workload.INGEST, 42, 5), Corpus.payloads(Workload.INGEST, 43, 5));
    }

    @Test
    void payloadsMatchTheirWorkload() {
        List<Corpus.Payload> rerank = Corpus.payloads(Workload.RERANK, 1, 3);
        assertEquals(32, rerank.getFirst().texts().size());
        assertTrue(rerank.getFirst().query().split(" ").length >= 6);

        Corpus.Payload query = Corpus.payloads(Workload.QUERY, 1, 1).getFirst();
        assertNull(query.query());
        assertEquals(1, query.texts().size());
        int words = query.texts().getFirst().split(" ").length;
        assertTrue(words >= 6 && words <= 16, "words: " + words);
    }
}

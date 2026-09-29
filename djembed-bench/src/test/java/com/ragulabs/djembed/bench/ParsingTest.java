package com.ragulabs.djembed.bench;

import org.junit.jupiter.api.Test;

import java.net.URI;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ParsingTest {

    @Test
    void prometheusSumsMatchingSeries() {
        Prometheus p = Prometheus.parse("""
                # HELP djembed_tokens_total Token slots
                # TYPE djembed_tokens_total counter
                djembed_tokens_total{kind="padding",model="bge-m3"} 15.0
                djembed_tokens_total{kind="real",model="bge-m3"} 25.0
                djembed_tokens_total{kind="real",model="other"} 7.0
                jvm_gc_pause_seconds_sum{action="end of minor GC",cause="Allocation Failure",gc="G1 Young Generation"} 0.5
                jvm_gc_pause_seconds_sum{action="end of major GC",cause="System.gc()",gc="G1 Old Generation"} 0.25
                jvm_gc_memory_allocated_bytes_total 1.0E9
                """);

        assertEquals(25.0, p.sum("djembed_tokens_total", "model=\"bge-m3\"", "kind=\"real\""));
        assertEquals(47.0, p.sum("djembed_tokens_total"));
        assertEquals(0.75, p.sum("jvm_gc_pause_seconds_sum"));
        assertEquals(1e9, p.sum("jvm_gc_memory_allocated_bytes_total"));
        assertEquals(0.0, p.sum("jvm_gc_pause_seconds"));
    }

    @Test
    void readsBothRerankFormatsInDocumentOrder() throws Exception {
        Target djembed = new Target.Djembed(URI.create("http://x"), null, "e", "r");
        Target tei = new Target.Tei(null, URI.create("http://y"), "e");
        Target infinity = new Target.Infinity(URI.create("http://z"), "e", "r");

        assertArrayEquals(new float[]{0.1f, 0.9f}, djembed.scores(Target.JSON.readTree("""
                {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}"""), 2));
        assertArrayEquals(new float[]{0.1f, 0.9f}, tei.scores(Target.JSON.readTree("""
                [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]"""), 2));
        assertArrayEquals(new float[]{0.1f, 0.9f}, infinity.scores(Target.JSON.readTree("""
                {"object": "rerank", "results": [{"relevance_score": 0.9, "index": 1}, {"relevance_score": 0.1, "index": 0}]}"""), 2));
    }

    @Test
    void readsOpenAiEmbeddingsByIndex() throws Exception {
        float[][] vectors = Target.embeddings(Target.JSON.readTree("""
                {"data": [{"index": 1, "embedding": [3, 4]}, {"index": 0, "embedding": [1, 2]}]}"""));

        assertArrayEquals(new float[]{1, 2}, vectors[0]);
        assertArrayEquals(new float[]{3, 4}, vectors[1]);
    }
}

package com.ragulabs.djembed.bench;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BenchArgsTest {

    @Test
    void parsesTargetsAndOptions() {
        BenchArgs args = BenchArgs.parse(new String[]{
                "--djembed", "http://gpu:8080/", "--tei-embed", "http://gpu:8081", "--workloads", "ingest,rerank",
                "--concurrency", "4,16", "--warmup", "500ms", "--duration", "2m", "--gpu", "0"});

        assertEquals("http://gpu:8080", args.djembed().toString());
        assertEquals(List.of(Workload.INGEST, Workload.RERANK), args.workloads());
        assertEquals(List.of(4, 16), args.concurrency());
        assertEquals(Duration.ofMillis(500), args.warmup());
        assertEquals(Duration.ofMinutes(2), args.duration());
        assertEquals(0, args.gpu());
        assertEquals("bge-m3", args.embedModel());

        List<Target> targets = args.targets();
        assertInstanceOf(Target.Djembed.class, targets.get(0));
        Target.Tei tei = assertInstanceOf(Target.Tei.class, targets.get(1));
        assertNull(tei.rerankBase());
        assertEquals(false, tei.supports(Workload.RERANK));
    }

    @Test
    void infinityServesBothModelsUnderTheirHubIds() {
        BenchArgs args = BenchArgs.parse(new String[]{"--djembed", "http://gpu:8080", "--infinity", "http://gpu:8083/"});

        Target.Infinity infinity = assertInstanceOf(Target.Infinity.class, args.targets().get(1));
        assertEquals("http://gpu:8083", infinity.base().toString());
        assertEquals("BAAI/bge-m3", infinity.embedModel());
        assertEquals("BAAI/bge-reranker-v2-m3", infinity.rerankModel());
        assertEquals(true, infinity.supports(Workload.RERANK));
        assertEquals("/embeddings", infinity.embed(List.of("a")).uri().getPath());
        assertEquals("/rerank", infinity.rerank("q", List.of("a")).uri().getPath());
    }

    @Test
    void rejectsBadInput() {
        assertThrows(IllegalArgumentException.class, () -> BenchArgs.parse(new String[]{}));
        assertThrows(IllegalArgumentException.class, () -> BenchArgs.parse(new String[]{"--djembed"}));
        assertThrows(IllegalArgumentException.class, () -> BenchArgs.parse(new String[]{"--nope", "x"}));
        assertThrows(IllegalArgumentException.class, () -> BenchArgs.parse(new String[]{"--djembed", "http://a", "--warmup", "5h"}));
    }
}

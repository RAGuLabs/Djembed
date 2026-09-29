package com.ragulabs.djembed.core;

import com.ragulabs.djembed.core.internal.OnnxGraph;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Graph facts of the benchmark's packed models, when present under {@code -Pmodels}.
 */
@Tag("onnx")
class RealGraphsTest {

    @Test
    void packedModelsAreDetectedWithTheirPrecision() {
        for (String name : new String[]{"bge-m3-fp16-packed", "bge-reranker-v2-m3-fp16-packed",
                "bge-m3-fp32-packed", "bge-reranker-v2-m3-fp32-packed"}) {
            Path dir = TestModels.model(name);
            Path onnx = Files.exists(dir.resolve("model.onnx")) ? dir.resolve("model.onnx") : dir.resolve("onnx/model.onnx");
            OnnxGraph graph = OnnxGraph.read(onnx);

            assertEquals(OnnxGraph.Attention.PACKED, graph.attention(), name);
            assertEquals(name.contains("fp16"), graph.halfPrecision(), name);
        }
    }
}

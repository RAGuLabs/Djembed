package com.ragulabs.djembed.core.internal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxGraphTest {

    @TempDir
    Path dir;

    @Test
    void countsOperators() throws IOException {
        byte[] model = model(FLOAT,
                node("MatMul", ""), node("MatMul", ""), node("Attention", "com.microsoft"), node("Softmax", "ai.onnx"));

        OnnxGraph graph = OnnxGraph.parse(model);

        assertEquals(2, graph.count("MatMul"));
        assertEquals(1, graph.count("Softmax"));
        assertEquals(1, graph.count("com.microsoft.Attention"));
        assertEquals(OnnxGraph.Attention.FUSED, graph.attention());
    }

    @Test
    void precisionFollowsTheWeights() throws IOException {
        assertTrue(OnnxGraph.parse(model(FLOAT16, node("MatMul", ""))).halfPrecision());
        assertFalse(OnnxGraph.parse(model(FLOAT, node("MatMul", ""))).halfPrecision());
        assertFalse(OnnxGraph.UNREADABLE.halfPrecision());
    }

    @Test
    void classifiesAttention() throws IOException {
        assertEquals(OnnxGraph.Attention.PACKED, OnnxGraph.parse(model(FLOAT,
                node("RemovePadding", "com.microsoft"), node("PackedAttention", "com.microsoft"))).attention());
        assertEquals(OnnxGraph.Attention.UNFUSED, OnnxGraph.parse(model(FLOAT, node("Softmax", ""))).attention());
    }

    @Test
    void unreadableFilesAreUnknownNotFatal() throws IOException {
        Path broken = Files.write(dir.resolve("model.onnx"), new byte[]{0x3a, (byte) 0xff});

        assertEquals(OnnxGraph.Attention.UNKNOWN, OnnxGraph.read(broken).attention());
    }

    @Test
    void readsFromFile() throws IOException {
        Path file = Files.write(dir.resolve("model.onnx"), model(FLOAT16, node("PackedAttention", "com.microsoft")));

        OnnxGraph graph = OnnxGraph.read(file);
        assertEquals(OnnxGraph.Attention.PACKED, graph.attention());
        assertTrue(graph.halfPrecision());
    }

    private static final int FLOAT = 1;
    private static final int FLOAT16 = 10;

    /** ModelProto { ir_version, producer_name, graph { nodes..., two weights of {@code weightType}, one int64 } }. */
    private static byte[] model(int weightType, byte[]... nodes) throws IOException {
        ByteArrayOutputStream graph = new ByteArrayOutputStream();
        for (byte[] node : nodes) {
            field(graph, 1, node);
        }
        field(graph, 5, weight(weightType, 4096));
        field(graph, 5, weight(weightType, 4096));
        field(graph, 5, weight(7, 8));

        ByteArrayOutputStream model = new ByteArrayOutputStream();
        model.write(new byte[]{0x08, 0x08});           // ir_version = 8 (varint field 1)
        field(model, 2, "test".getBytes(StandardCharsets.UTF_8));
        field(model, 7, graph.toByteArray());
        return model.toByteArray();
    }

    /** TensorProto { dims, data_type, name, raw_data }: the reader must skip the data. */
    private static byte[] weight(int dataType, int bytes) throws IOException {
        ByteArrayOutputStream tensor = new ByteArrayOutputStream();
        tensor.write(0x08);                              // dims (varint field 1)
        varint(tensor, bytes);
        tensor.write(0x10);                              // data_type (varint field 2)
        varint(tensor, dataType);
        field(tensor, 8, "w".getBytes(StandardCharsets.UTF_8));
        field(tensor, 9, new byte[bytes]);
        return tensor.toByteArray();
    }

    private static byte[] node(String op, String domain) throws IOException {
        ByteArrayOutputStream node = new ByteArrayOutputStream();
        field(node, 1, "in".getBytes(StandardCharsets.UTF_8));
        field(node, 4, op.getBytes(StandardCharsets.UTF_8));
        if (!domain.isEmpty()) {
            field(node, 7, domain.getBytes(StandardCharsets.UTF_8));
        }
        return node.toByteArray();
    }

    private static void field(ByteArrayOutputStream out, int number, byte[] value) throws IOException {
        varint(out, (long) number << 3 | 2);
        varint(out, value.length);
        out.write(value);
    }

    private static void varint(ByteArrayOutputStream out, long value) {
        while ((value & ~0x7FL) != 0) {
            out.write((int) (value & 0x7F) | 0x80);
            value >>>= 7;
        }
        out.write((int) value);
    }
}

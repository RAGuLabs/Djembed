package com.ragulabs.djembed.core.internal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OnnxGraphTest {

    @TempDir
    Path dir;

    @Test
    void countsOperatorsUpToTheFirstWeight() throws IOException {
        byte[] model = model(
                node("MatMul", ""), node("MatMul", ""), node("Attention", "com.microsoft"), node("Softmax", "ai.onnx"));

        OnnxGraph graph = OnnxGraph.parse(model);

        assertEquals(2, graph.count("MatMul"));
        assertEquals(1, graph.count("Softmax"));
        assertEquals(1, graph.count("com.microsoft.Attention"));
        assertEquals(0, graph.count("Gather"), "nodes after the first initializer are never read");
        assertEquals(OnnxGraph.Attention.FUSED, graph.attention());
    }

    @Test
    void classifiesAttention() throws IOException {
        assertEquals(OnnxGraph.Attention.PACKED, OnnxGraph.parse(model(
                node("RemovePadding", "com.microsoft"), node("PackedAttention", "com.microsoft"))).attention());
        assertEquals(OnnxGraph.Attention.UNFUSED, OnnxGraph.parse(model(node("Softmax", ""))).attention());
    }

    @Test
    void unreadableFilesAreUnknownNotFatal() throws IOException {
        Path broken = Files.write(dir.resolve("model.onnx"), new byte[]{0x3a, (byte) 0xff});

        assertEquals(OnnxGraph.Attention.UNKNOWN, OnnxGraph.read(broken).attention());
    }

    @Test
    void readsFromFile() throws IOException {
        Path file = Files.write(dir.resolve("model.onnx"), model(node("PackedAttention", "com.microsoft")));

        assertEquals(OnnxGraph.Attention.PACKED, OnnxGraph.read(file).attention());
    }

    /** ModelProto { ir_version, producer_name, graph { nodes..., initializer, node Gather } }. */
    private static byte[] model(byte[]... nodes) throws IOException {
        ByteArrayOutputStream graph = new ByteArrayOutputStream();
        for (byte[] node : nodes) {
            field(graph, 1, node);
        }
        field(graph, 5, new byte[64]);                 // an initializer: reading stops here
        field(graph, 1, node("Gather", ""));

        ByteArrayOutputStream model = new ByteArrayOutputStream();
        model.write(new byte[]{0x08, 0x08});           // ir_version = 8 (varint field 1)
        field(model, 2, "test".getBytes(StandardCharsets.UTF_8));
        field(model, 7, graph.toByteArray());
        return model.toByteArray();
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

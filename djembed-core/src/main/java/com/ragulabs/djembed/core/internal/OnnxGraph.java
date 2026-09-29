package com.ragulabs.djembed.core.internal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Operator counts and weight precision of an ONNX model's main graph, read straight from the protobuf.
 *
 * <p>Only headers are read: node definitions, and the {@code data_type} of each weight, whose data is skipped (a
 * seek on disk), so even a gigabyte single-file model costs a few kilobytes of reading. What the graph reveals decides
 * how batches are planned: a graph in ONNX Runtime's packing mode computes no padding in its encoder, and whether its
 * weights are half precision decides which attention kernels it gets.
 */
public final class OnnxGraph {

    private static final Logger log = LoggerFactory.getLogger(OnnxGraph.class);

    private static final String MS = "com.microsoft.";

    /** How the graph computes attention. */
    public enum Attention {
        /** {@code PackedAttention}: the encoder runs on real tokens only, padding removed. */
        PACKED,
        /** Fused {@code Attention}/{@code MultiHeadAttention}: fast kernels over padded batches. */
        FUSED,
        /** Attention spelled out in elementary operators. */
        UNFUSED,
        /** The graph could not be read. */
        UNKNOWN
    }

    static final OnnxGraph UNREADABLE = new OnnxGraph(Map.of(), Map.of(), false);

    // TensorProto.DataType values.
    private static final int FLOAT = 1;
    private static final int FLOAT16 = 10;
    private static final int BFLOAT16 = 16;

    private final Map<String, Integer> operators;
    private final Map<Integer, Integer> weightTypes;
    private final boolean readable;

    private OnnxGraph(Map<String, Integer> operators, Map<Integer, Integer> weightTypes, boolean readable) {
        this.operators = operators;
        this.weightTypes = weightTypes;
        this.readable = readable;
    }

    /** Reads {@code file}; an unreadable graph yields {@link Attention#UNKNOWN} rather than an error. */
    public static OnnxGraph read(Path file) {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(file), 1 << 16)) {
            return read(new ProtoReader(in));
        } catch (IOException | RuntimeException e) {
            log.warn("Cannot read the operators of {}: {}", file, e.toString());
            return UNREADABLE;
        }
    }

    /** Parses bytes already in memory; for tests. */
    static OnnxGraph parse(byte[] model) throws IOException {
        return read(new ProtoReader(new ByteArrayInputStream(model)));
    }

    /** Occurrences of {@code operator}, domain-qualified outside the default domain (e.g. {@code com.microsoft.Attention}). */
    public int count(String operator) {
        return operators.getOrDefault(operator, 0);
    }

    public Attention attention() {
        if (!readable) {
            return Attention.UNKNOWN;
        }
        if (count(MS + "PackedAttention") + count(MS + "PackedMultiHeadAttention") > 0) {
            return Attention.PACKED;
        }
        if (count(MS + "Attention") + count(MS + "MultiHeadAttention") > 0) {
            return Attention.FUSED;
        }
        return Attention.UNFUSED;
    }

    /**
     * Whether the graph computes in half precision: its float16/bfloat16 weights outnumber its float32 ones. ONNX
     * Runtime's fp16 conversion stores every floating-point weight as float16, while a fp32 graph has none.
     */
    public boolean halfPrecision() {
        int half = weightTypes.getOrDefault(FLOAT16, 0) + weightTypes.getOrDefault(BFLOAT16, 0);
        return half > weightTypes.getOrDefault(FLOAT, 0);
    }

    // ModelProto.graph = 7; GraphProto.node = 1, GraphProto.initializer = 5; NodeProto.op_type = 4,
    // NodeProto.domain = 7; TensorProto.data_type = 2.
    private static final int MODEL_GRAPH = 7;
    private static final int GRAPH_NODE = 1;
    private static final int GRAPH_INITIALIZER = 5;
    private static final int NODE_OP_TYPE = 4;
    private static final int NODE_DOMAIN = 7;
    private static final int TENSOR_DATA_TYPE = 2;

    private static OnnxGraph read(ProtoReader model) throws IOException {
        Map<String, Integer> operators = new HashMap<>();
        Map<Integer, Integer> weightTypes = new HashMap<>();
        while (!model.atEnd()) {
            long key = model.varint();
            if (field(key) != MODEL_GRAPH || wire(key) != ProtoReader.LENGTH_DELIMITED) {
                model.skip(wire(key));
                continue;
            }
            long end = model.position() + model.varint();
            while (model.position() < end) {
                long graphKey = model.varint();
                if (field(graphKey) == GRAPH_NODE && wire(graphKey) == ProtoReader.LENGTH_DELIMITED) {
                    operators.merge(operator(model.bytes((int) model.varint())), 1, Integer::sum);
                } else if (field(graphKey) == GRAPH_INITIALIZER && wire(graphKey) == ProtoReader.LENGTH_DELIMITED) {
                    int type = weightType(model, model.position() + model.varint());
                    if (type > 0) {
                        weightTypes.merge(type, 1, Integer::sum);
                    }
                } else {
                    model.skip(wire(graphKey));
                }
            }
            break;
        }
        return new OnnxGraph(Collections.unmodifiableMap(operators), Collections.unmodifiableMap(weightTypes), true);
    }

    /** The {@code data_type} of the weight ending at {@code end}, skipping its data; 0 when absent. */
    private static int weightType(ProtoReader model, long end) throws IOException {
        int type = 0;
        while (model.position() < end) {
            long key = model.varint();
            if (field(key) == TENSOR_DATA_TYPE && wire(key) == ProtoReader.VARINT) {
                type = (int) model.varint();
            } else {
                model.skip(wire(key));
            }
        }
        return type;
    }

    private static String operator(byte[] node) throws IOException {
        ProtoReader reader = new ProtoReader(new ByteArrayInputStream(node));
        String op = "";
        String domain = "";
        while (!reader.atEnd()) {
            long key = reader.varint();
            if (field(key) == NODE_OP_TYPE && wire(key) == ProtoReader.LENGTH_DELIMITED) {
                op = new String(reader.bytes((int) reader.varint()), StandardCharsets.UTF_8);
            } else if (field(key) == NODE_DOMAIN && wire(key) == ProtoReader.LENGTH_DELIMITED) {
                domain = new String(reader.bytes((int) reader.varint()), StandardCharsets.UTF_8);
            } else {
                reader.skip(wire(key));
            }
        }
        return domain.isEmpty() || domain.equals("ai.onnx") ? op : domain + "." + op;
    }

    private static int field(long key) {
        return (int) (key >>> 3);
    }

    private static int wire(long key) {
        return (int) (key & 7);
    }

    /** Sequential protobuf wire-format reading over a stream; large fields are skipped, not read. */
    private static final class ProtoReader {

        static final int VARINT = 0;
        static final int FIXED64 = 1;
        static final int LENGTH_DELIMITED = 2;
        static final int FIXED32 = 5;

        private final InputStream in;
        private long position;
        private int peeked = -2;

        ProtoReader(InputStream in) {
            this.in = in;
        }

        long position() {
            return position;
        }

        boolean atEnd() throws IOException {
            if (peeked == -2) {
                peeked = in.read();
            }
            return peeked < 0;
        }

        private int next() throws IOException {
            int b;
            if (peeked != -2) {
                b = peeked;
                peeked = -2;
            } else {
                b = in.read();
            }
            if (b < 0) {
                throw new EOFException("truncated protobuf at byte " + position);
            }
            position++;
            return b;
        }

        long varint() throws IOException {
            long value = 0;
            for (int shift = 0; shift < 64; shift += 7) {
                int b = next();
                value |= (long) (b & 0x7f) << shift;
                if ((b & 0x80) == 0) {
                    return value;
                }
            }
            throw new IOException("malformed varint at byte " + position);
        }

        byte[] bytes(int length) throws IOException {
            byte[] out = new byte[length];
            int from = 0;
            if (length > 0 && peeked >= 0) {
                out[0] = (byte) next();
                from = 1;
            }
            int read = in.readNBytes(out, from, length - from);
            if (read != length - from) {
                throw new EOFException("truncated protobuf at byte " + position);
            }
            position += read;
            return out;
        }

        void skip(int wireType) throws IOException {
            switch (wireType) {
                case VARINT -> varint();
                case FIXED64 -> skipBytes(8);
                case LENGTH_DELIMITED -> skipBytes(varint());
                case FIXED32 -> skipBytes(4);
                default -> throw new IOException("unsupported wire type " + wireType + " at byte " + position);
            }
        }

        private void skipBytes(long length) throws IOException {
            long remaining = length;
            if (remaining > 0 && peeked >= 0) {
                next();
                remaining--;
            }
            in.skipNBytes(remaining);
            position += remaining;
        }
    }
}

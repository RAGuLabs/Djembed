package dev.fgnm.djembed.core.internal;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.TensorInfo;
import dev.fgnm.djembed.core.DjembedException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Off-heap input and output buffers of one model, allocated once and reused by every forward pass.
 *
 * <p>Inputs are assembled with bulk copies straight into native memory, and ONNX Runtime reads them in place
 * (tensors over native-order direct buffers are not copied). The output tensor is pre-allocated over native
 * memory too and handed to the session as a pinned output, so results land here without an intermediate heap
 * array. A forward pass allocates only a handful of small wrapper objects, whatever the batch size.
 *
 * <p>Not thread-safe: the owning engine serialises access.
 */
public final class Workspace implements AutoCloseable {

    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT;
    private static final ValueLayout.OfShort HALF = ValueLayout.JAVA_SHORT;

    private final OnnxModel model;
    private final OrtEnvironment env;
    private final Arena arena;

    private final int tokenCapacity;
    private final int maxRowLength;
    private final int maxRows;

    private final MemorySegment ids;
    private final MemorySegment mask;
    private final MemorySegment typeIds;
    private final MemorySegment padRow;
    private final MemorySegment onesRow;
    private final MemorySegment zerosRow;
    private final MemorySegment output;

    private final String outputName;
    private final boolean halfOutput;
    private final boolean perTokenOutput;
    private final int outputWidth;
    private final Map<String, OnnxTensorLike> inputs = new HashMap<>(4);

    private int rows;
    private int rowLength;

    /**
     * @param outputName    the only output the session computes and copies back
     * @param outputWidth   last dimension of that output
     * @param tokenCapacity largest {@code rows × row length} a forward pass may use; at least {@code maxRowLength}
     * @param maxRowLength  longest sequence, special tokens included
     * @param maxRows       most rows of a forward pass
     * @param padTokenId    id written after the end of shorter rows
     */
    public Workspace(OnnxModel model, String outputName, int outputWidth, int tokenCapacity, int maxRowLength,
                     int maxRows, long padTokenId) {
        if (tokenCapacity < maxRowLength) {
            throw new IllegalArgumentException("tokenCapacity " + tokenCapacity + " cannot hold one row of " + maxRowLength);
        }
        TensorInfo outputInfo = model.outputs().get(outputName);
        if (outputInfo == null) {
            throw new DjembedException("Model has no output '" + outputName + "'");
        }
        int rank = outputInfo.getShape().length;
        if (rank != 2 && rank != 3) {
            throw new DjembedException("Output '" + outputName + "' must be [batch, width] or [batch, tokens, width], found " + outputInfo);
        }
        this.halfOutput = switch (outputInfo.type) {
            case FLOAT -> false;
            case FLOAT16 -> true;
            default -> throw new DjembedException("Output '" + outputName + "' must be float32 or float16, found " + outputInfo.type);
        };
        this.model = model;
        this.env = OnnxModel.environment();
        this.outputName = outputName;
        this.perTokenOutput = rank == 3;
        this.outputWidth = outputWidth;
        this.tokenCapacity = tokenCapacity;
        this.maxRowLength = maxRowLength;
        this.maxRows = maxRows;

        this.arena = Arena.ofShared();
        this.ids = arena.allocate(LONG, tokenCapacity);
        this.mask = arena.allocate(LONG, tokenCapacity);
        this.typeIds = model.takesTokenTypeIds() ? arena.allocate(LONG, tokenCapacity) : null;
        this.padRow = arena.allocate(LONG, maxRowLength);
        this.onesRow = arena.allocate(LONG, maxRowLength);
        this.zerosRow = arena.allocate(LONG, maxRowLength);
        padRow.elements(LONG).forEach(e -> e.set(LONG, 0, padTokenId));
        onesRow.elements(LONG).forEach(e -> e.set(LONG, 0, 1L));

        long outputElements = (long) (perTokenOutput ? tokenCapacity : maxRows) * outputWidth;
        this.output = arena.allocate(halfOutput ? HALF : FLOAT, outputElements);
    }

    /** Starts a forward pass of {@code rows} sequences padded to {@code rowLength} tokens. */
    public void begin(int rows, int rowLength) {
        if (rows < 1 || rows > maxRows || rowLength < 1 || rowLength > maxRowLength
                || (long) rows * rowLength > tokenCapacity) {
            throw new IllegalStateException("Batch " + rows + "×" + rowLength + " exceeds the workspace ("
                    + maxRows + " rows, " + maxRowLength + " tokens per row, " + tokenCapacity + " tokens)");
        }
        this.rows = rows;
        this.rowLength = rowLength;
    }

    /**
     * Writes row {@code row}: {@code prefix} (may be null), {@code source[from .. to)}, {@code suffix} (may be null),
     * then padding up to the batch row length.
     *
     * @param types token type ids aligned with {@code source}, or {@code null} for all zeros
     */
    public void putRow(int row, long[] source, int from, int to, long[] prefix, long[] suffix, long[] types) {
        long rowStart = (long) row * rowLength * Long.BYTES;
        long at = rowStart;
        if (prefix != null) {
            MemorySegment.copy(prefix, 0, ids, LONG, at, prefix.length);
            at += (long) prefix.length * Long.BYTES;
        }
        MemorySegment.copy(source, from, ids, LONG, at, to - from);
        at += (long) (to - from) * Long.BYTES;
        if (suffix != null) {
            MemorySegment.copy(suffix, 0, ids, LONG, at, suffix.length);
            at += (long) suffix.length * Long.BYTES;
        }

        long used = at - rowStart;
        long padding = (long) rowLength * Long.BYTES - used;
        if (padding < 0) {
            throw new IllegalStateException("Row " + row + " holds " + used / Long.BYTES + " tokens, more than the batch row length " + rowLength);
        }
        MemorySegment.copy(padRow, 0, ids, at, padding);
        MemorySegment.copy(onesRow, 0, mask, rowStart, used);
        MemorySegment.copy(zerosRow, 0, mask, at, padding);

        if (typeIds != null) {
            MemorySegment.copy(zerosRow, 0, typeIds, rowStart, used + padding);
            if (types != null) {
                long typeStart = rowStart + (prefix != null ? (long) prefix.length * Long.BYTES : 0);
                MemorySegment.copy(types, from, typeIds, LONG, typeStart, to - from);
            }
        }
    }

    /** Runs the forward pass over the rows written since {@link #begin}. */
    public void run() {
        long tokens = (long) rows * rowLength;
        long[] inputShape = {rows, rowLength};
        long[] outputShape = perTokenOutput ? new long[]{rows, rowLength, outputWidth} : new long[]{rows, outputWidth};
        long outputElements = perTokenOutput ? tokens * outputWidth : (long) rows * outputWidth;

        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, ids.asSlice(0, tokens * Long.BYTES).asByteBuffer()
                     .order(ByteOrder.nativeOrder()).asLongBuffer(), inputShape);
             OnnxTensor maskTensor = OnnxTensor.createTensor(env, mask.asSlice(0, tokens * Long.BYTES).asByteBuffer()
                     .order(ByteOrder.nativeOrder()).asLongBuffer(), inputShape);
             OnnxTensor typesTensor = typeIds == null ? null : OnnxTensor.createTensor(env, typeIds.asSlice(0, tokens * Long.BYTES)
                     .asByteBuffer().order(ByteOrder.nativeOrder()).asLongBuffer(), inputShape);
             OnnxTensor outputTensor = outputTensor(outputElements, outputShape)) {
            inputs.clear();
            inputs.put(OnnxModel.INPUT_IDS, idsTensor);
            inputs.put(OnnxModel.ATTENTION_MASK, maskTensor);
            if (typesTensor != null) {
                inputs.put(OnnxModel.TOKEN_TYPE_IDS, typesTensor);
            }
            // Pinning the output makes it the only one computed and copied back, with no extra requested outputs.
            // The result only borrows the pinned tensor; closing it releases nothing we still read.
            model.session().run(inputs, Set.of(), Map.of(outputName, outputTensor)).close();
        } catch (OrtException e) {
            throw new DjembedException("Inference failed on a " + rows + "×" + rowLength + " batch: " + e.getMessage(), e);
        } finally {
            inputs.clear();
        }
    }

    private OnnxTensor outputTensor(long elements, long[] shape) throws OrtException {
        if (halfOutput) {
            return OnnxTensor.createTensor(env, output.asSlice(0, elements * Short.BYTES).asByteBuffer()
                    .order(ByteOrder.nativeOrder()).asShortBuffer(), shape, OnnxJavaType.FLOAT16);
        }
        return OnnxTensor.createTensor(env, output.asSlice(0, elements * Float.BYTES).asByteBuffer()
                .order(ByteOrder.nativeOrder()).asFloatBuffer(), shape);
    }

    /** Element {@code index} of the last output, row-major, widened to float32. */
    public float output(long index) {
        return halfOutput
                ? Float.float16ToFloat(output.getAtIndex(HALF, index))
                : output.getAtIndex(FLOAT, index);
    }

    /** Index of {@code output[row][token][0]} for per-token outputs. */
    public long tokenOffset(int row, int token) {
        return ((long) row * rowLength + token) * outputWidth;
    }

    /** Index of {@code output[row][0]} for per-row outputs. */
    public long rowOffset(int row) {
        return (long) row * outputWidth;
    }

    @Override
    public void close() {
        arena.close();
    }
}

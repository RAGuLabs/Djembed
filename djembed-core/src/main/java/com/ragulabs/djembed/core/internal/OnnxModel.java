package com.ragulabs.djembed.core.internal;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import com.ragulabs.djembed.core.Device;
import com.ragulabs.djembed.core.DjembedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * An ONNX Runtime session for an encoder that takes {@code input_ids}, {@code attention_mask} and optionally
 * {@code token_type_ids}.
 */
public final class OnnxModel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxModel.class);

    static final String INPUT_IDS = "input_ids";
    static final String ATTENTION_MASK = "attention_mask";
    static final String TOKEN_TYPE_IDS = "token_type_ids";

    private final OrtSession session;
    private final boolean tokenTypeIds;

    private OnnxModel(OrtSession session) throws OrtException {
        this.session = session;
        Map<String, NodeInfo> inputs = session.getInputInfo();
        for (Map.Entry<String, NodeInfo> input : inputs.entrySet()) {
            String name = input.getKey();
            if (!name.equals(INPUT_IDS) && !name.equals(ATTENTION_MASK) && !name.equals(TOKEN_TYPE_IDS)) {
                throw new DjembedException("Unsupported model input '" + name + "': expected "
                        + INPUT_IDS + ", " + ATTENTION_MASK + " and optionally " + TOKEN_TYPE_IDS);
            }
            if (!(input.getValue().getInfo() instanceof TensorInfo info) || info.type != OnnxJavaType.INT64) {
                throw new DjembedException("Model input '" + name + "' must be an int64 tensor, found " + input.getValue().getInfo());
            }
        }
        if (!inputs.containsKey(INPUT_IDS) || !inputs.containsKey(ATTENTION_MASK)) {
            throw new DjembedException("Model must take " + INPUT_IDS + " and " + ATTENTION_MASK + ", found " + inputs.keySet());
        }
        this.tokenTypeIds = inputs.containsKey(TOKEN_TYPE_IDS);
    }

    /**
     * @param tf32 on CUDA, whether float32 matrix multiplications may use TensorFloat-32 tensor cores
     */
    public static OnnxModel load(Path file, Device device, boolean tf32) {
        long started = System.nanoTime();
        // The environment registers ONNX Runtime's logger, which provider options already need.
        OrtEnvironment env = Environment.ORT;
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            OnnxModel model = switch (device) {
                case Device.Cpu _ -> new OnnxModel(env.createSession(file.toString(), options));
                case Device.Cuda(int deviceId) -> {
                    try (OrtCUDAProviderOptions cuda = new OrtCUDAProviderOptions(deviceId)) {
                        // Grow the device arena by what each request needs rather than by powers of two:
                        // input shapes change on every batch, and doubling strands most of the memory.
                        cuda.add("arena_extend_strategy", "kSameAsRequested");
                        cuda.add("use_tf32", tf32 ? "1" : "0");
                        options.addCUDA(cuda);
                        yield new OnnxModel(env.createSession(file.toString(), options));
                    }
                }
            };
            log.info("Loaded {} on {}{} in {} ms", file, device,
                    device instanceof Device.Cuda ? (tf32 ? " (tf32)" : " (strict fp32)") : "",
                    (System.nanoTime() - started) / 1_000_000);
            return model;
        } catch (OrtException e) {
            throw new DjembedException("Cannot load ONNX model " + file + " on " + device + ": " + e.getMessage(), e);
        }
    }

    OrtSession session() {
        return session;
    }

    public boolean takesTokenTypeIds() {
        return tokenTypeIds;
    }

    /** Output tensors by name. */
    public Map<String, TensorInfo> outputs() {
        try {
            Map<String, TensorInfo> outputs = new LinkedHashMap<>();
            for (Map.Entry<String, NodeInfo> e : session.getOutputInfo().entrySet()) {
                if (e.getValue().getInfo() instanceof TensorInfo info) {
                    outputs.put(e.getKey(), info);
                }
            }
            return outputs;
        } catch (OrtException e) {
            throw new DjembedException("Cannot read model outputs: " + e.getMessage(), e);
        }
    }

    static OrtEnvironment environment() {
        return Environment.ORT;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            log.warn("Error closing ONNX session", e);
        }
    }

    /**
     * Created on first use, so merely loading these classes does not start ONNX Runtime, and the telemetry switch
     * is in place before its native library initialises. Every ONNX Runtime call in Djembed goes through here first.
     */
    private static final class Environment {

        static final OrtEnvironment ORT = create();

        private static OrtEnvironment create() {
            Telemetry.disableBeforeOnnxRuntimeLoads();
            OrtEnvironment env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "djembed");
            try {
                env.setTelemetry(false);
            } catch (OrtException e) {
                log.warn("Cannot disable ONNX Runtime telemetry through the API", e);
            }
            return env;
        }
    }
}

package dev.fgnm.djembed.core.internal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The build deliberately does not export ORT_DISABLE_TELEMETRY to test JVMs, so this checks Djembed's own guard.
 */
class TelemetryTest {

    @Test
    void onnxRuntimeStartsWithTelemetryDisabled() {
        OnnxModel.environment();

        assertEquals("1", Telemetry.nativeValue());
    }
}

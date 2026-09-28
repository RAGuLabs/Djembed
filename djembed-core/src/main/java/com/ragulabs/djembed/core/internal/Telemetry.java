package com.ragulabs.djembed.core.internal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Optional;

/**
 * Keeps ONNX Runtime from collecting telemetry.
 *
 * <p>ONNX Runtime reads {@code ORT_DISABLE_TELEMETRY} while its native library initialises, before any Java API
 * call could switch telemetry off, and it collects process data at that point (1.29 even segfaults the JVM on long
 * command lines while doing so). Since Java cannot change its own environment, the variable is written into the
 * native process environment through libc {@code setenv} before ONNX Runtime is first touched. Operators need not
 * remember to export it, and telemetry stays off even when Djembed is embedded as a library.
 */
final class Telemetry {

    private static final Logger log = LoggerFactory.getLogger(Telemetry.class);

    static final String VARIABLE = "ORT_DISABLE_TELEMETRY";

    private Telemetry() {
    }

    /** Must run before the first use of any ONNX Runtime class. */
    static void disableBeforeOnnxRuntimeLoads() {
        Optional<MethodHandle> setenv = setenv();
        if (setenv.isEmpty()) {
            log.warn("libc setenv not found: export {}=1 before starting the JVM to keep ONNX Runtime telemetry off", VARIABLE);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            int rc = (int) setenv.get().invokeExact(arena.allocateFrom(VARIABLE), arena.allocateFrom("1"), 1);
            if (rc != 0) {
                log.warn("setenv({}) failed with {}: export {}=1 before starting the JVM", VARIABLE, rc, VARIABLE);
            }
        } catch (Throwable e) {
            log.warn("Cannot set {}: export it before starting the JVM", VARIABLE, e);
        }
    }

    /** {@code int setenv(const char *name, const char *value, int overwrite)}, absent on Windows. */
    @SuppressWarnings("restricted")
    private static Optional<MethodHandle> setenv() {
        Linker linker = Linker.nativeLinker();
        SymbolLookup libc = linker.defaultLookup();
        return libc.find("setenv").map(symbol -> linker.downcallHandle(symbol,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)));
    }

    /** The value ONNX Runtime sees, read back from the native environment; {@code null} when unset or unreadable. */
    @SuppressWarnings("restricted")
    static String nativeValue() {
        Linker linker = Linker.nativeLinker();
        Optional<MemorySegment> getenv = linker.defaultLookup().find("getenv");
        if (getenv.isEmpty()) {
            return null;
        }
        MethodHandle handle = linker.downcallHandle(getenv.get(), FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment value = (MemorySegment) handle.invokeExact(arena.allocateFrom(VARIABLE));
            return value.equals(MemorySegment.NULL) ? null : value.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable e) {
            return null;
        }
    }
}

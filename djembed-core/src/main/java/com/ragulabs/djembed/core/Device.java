package com.ragulabs.djembed.core;

import java.util.Locale;

/**
 * Where a model runs.
 */
public sealed interface Device permits Device.Cpu, Device.Cuda {

    static Device cpu() {
        return Cpu.INSTANCE;
    }

    static Device cuda(int deviceId) {
        return new Cuda(deviceId);
    }

    /**
     * Parses {@code cpu}, {@code cuda} (device 0) or {@code cuda:N}.
     */
    static Device parse(String value) {
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.equals("cpu")) {
            return cpu();
        }
        if (v.equals("cuda")) {
            return cuda(0);
        }
        if (v.startsWith("cuda:")) {
            try {
                return cuda(Integer.parseInt(v.substring(5)));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid CUDA device: " + value, e);
            }
        }
        throw new IllegalArgumentException("Unknown device '" + value + "', expected cpu, cuda or cuda:N");
    }

    record Cpu() implements Device {

        private static final Cpu INSTANCE = new Cpu();

        @Override
        public String toString() {
            return "cpu";
        }
    }

    record Cuda(int deviceId) implements Device {

        public Cuda {
            if (deviceId < 0) {
                throw new IllegalArgumentException("CUDA device id must be >= 0: " + deviceId);
            }
        }

        @Override
        public String toString() {
            return "cuda:" + deviceId;
        }
    }
}

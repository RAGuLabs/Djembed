package com.ragulabs.djembed.bench;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Samples GPU utilisation and memory with {@code nvidia-smi} while a window is open. Meaningful only when the
 * benchmark runs on the machine that owns the GPU.
 */
final class GpuSampler {

    private final int index;
    private Process process;
    private Thread reader;
    private volatile double utilizationSum;
    private volatile long samples;
    private volatile long maxMemoryMib;

    GpuSampler(int index) {
        this.index = index;
    }

    record Sample(double utilization, long maxMemoryMib) {
    }

    void start() throws IOException {
        utilizationSum = 0;
        samples = 0;
        maxMemoryMib = 0;
        process = new ProcessBuilder("nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", "-i", Integer.toString(index), "-lms", "250")
                .redirectErrorStream(true)
                .start();
        reader = Thread.ofVirtual().start(() -> {
            try (BufferedReader lines = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = lines.readLine()) != null) {
                    String[] fields = line.split(",");
                    if (fields.length == 2) {
                        try {
                            utilizationSum += Double.parseDouble(fields[0].trim());
                            maxMemoryMib = Math.max(maxMemoryMib, Long.parseLong(fields[1].trim()));
                            samples++;
                        } catch (NumberFormatException ignored) {
                            // Header or warning line.
                        }
                    }
                }
            } catch (IOException ignored) {
                // Process stopped.
            }
        });
    }

    Sample stop() throws InterruptedException {
        process.destroy();
        reader.join();
        return new Sample(samples == 0 ? Double.NaN : utilizationSum / samples, maxMemoryMib);
    }

    /** {@code nvidia-smi -L} line for the GPU, or null when unavailable. */
    static String describe(int index) {
        try {
            Process p = new ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            p.waitFor();
            for (String line : out.split("\n")) {
                if (line.startsWith("GPU " + index + ":")) {
                    return line.replaceAll("\\s*\\(UUID.*\\)", "").trim();
                }
            }
        } catch (IOException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return null;
    }
}

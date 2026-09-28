package dev.fgnm.djembed.core;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Locates the checkpoints of the {@code onnx}-tagged tests: {@code -Pmodels=<folder holding one folder per model>}
 * and {@code -Pdevice=cpu|cuda:N}.
 */
final class TestModels {

    private TestModels() {
    }

    static Path model(String name) {
        String root = System.getProperty("djembed.test.models", "");
        assumeTrue(!root.isBlank(), "-Pmodels not set");
        Path dir = Path.of(root, name);
        assumeTrue(Files.isDirectory(dir), dir + " not found");
        return dir;
    }

    static EngineOptions engine() {
        return EngineOptions.defaults().withDevice(Device.parse(System.getProperty("djembed.test.device", "cpu")));
    }
}

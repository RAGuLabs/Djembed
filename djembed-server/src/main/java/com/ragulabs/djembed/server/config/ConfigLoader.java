package com.ragulabs.djembed.server.config;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.dataformat.yaml.YAMLMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class ConfigLoader {

    private static final YAMLMapper MAPPER = YAMLMapper.builder()
            .propertyNamingStrategy(PropertyNamingStrategies.KEBAB_CASE)
            .enable(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .build();

    private ConfigLoader() {
    }

    public static DjembedConfig load(Path file) {
        DjembedConfig raw = null;
        try {
            String content = Files.readString(file);
            if (!content.isBlank()) {
                raw = MAPPER.readValue(content, DjembedConfig.class);
            }
        } catch (IOException e) {
            throw new ConfigException("Cannot load configuration " + file.toAbsolutePath() + ": " + e.getMessage(), e);
        }
        // A blank file, or one holding only comments or `~`, means "all defaults".
        if (raw == null) {
            raw = new DjembedConfig(null, null);
        }

        Path baseDir = file.toAbsolutePath().getParent();
        return new DjembedConfig(
                raw.server(),
                raw.models().stream().map(m -> m.resolveAgainst(baseDir)).toList());
    }
}

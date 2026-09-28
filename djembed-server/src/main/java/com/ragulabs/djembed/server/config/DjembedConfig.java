package com.ragulabs.djembed.server.config;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Root of the YAML configuration file.
 */
public record DjembedConfig(ServerConfig server, List<ModelConfig> models) {

    public DjembedConfig {
        server = server != null ? server : ServerConfig.DEFAULT;
        models = models != null ? List.copyOf(models) : List.of();

        Set<String> names = new HashSet<>();
        for (ModelConfig model : models) {
            if (!names.add(model.name())) {
                throw new IllegalArgumentException("Duplicate model name: " + model.name());
            }
        }
    }
}

package com.ragulabs.djembed.core;

/**
 * @param engine     execution settings; inputs over the token limit are truncated, longest sequence first
 * @param activation transformation applied to the model logit
 */
public record RerankOptions(EngineOptions engine, ScoreActivation activation) {

    private static final RerankOptions DEFAULTS = new RerankOptions(EngineOptions.defaults(), ScoreActivation.SIGMOID);

    public RerankOptions {
        if (engine == null) {
            throw new IllegalArgumentException("engine options are required");
        }
        if (activation == null) {
            throw new IllegalArgumentException("activation is required");
        }
    }

    public static RerankOptions defaults() {
        return DEFAULTS;
    }

    public RerankOptions withEngine(EngineOptions engine) {
        return new RerankOptions(engine, activation);
    }

    public RerankOptions withActivation(ScoreActivation activation) {
        return new RerankOptions(engine, activation);
    }
}

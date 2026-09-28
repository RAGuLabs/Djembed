package com.ragulabs.djembed.core.internal;

import com.ragulabs.djembed.core.DjembedException;
import com.ragulabs.djembed.core.EngineOptions;

public final class Limits {

    private Limits() {
    }

    /** The configured token limit, or the model's own when {@link EngineOptions#MODEL_MAX}. Never above the model's. */
    public static int maxInputTokens(ModelDirectory model, EngineOptions options) {
        int modelMax = model.maxInputTokens();
        int requested = options.maxInputTokens();
        if (requested == EngineOptions.MODEL_MAX) {
            return modelMax;
        }
        if (requested > modelMax) {
            throw new DjembedException("maxInputTokens " + requested + " exceeds the limit of " + model.root() + " (" + modelMax + ")");
        }
        return requested;
    }

    /** Tokens the workspace must hold: the batch budget, and never less than one full-length row. */
    public static int tokenCapacity(EngineOptions options, int maxInputTokens) {
        return Math.max(options.tokenBudget(), maxInputTokens);
    }
}

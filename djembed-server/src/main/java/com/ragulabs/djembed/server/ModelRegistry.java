package com.ragulabs.djembed.server;

import com.ragulabs.djembed.core.EmbeddingEngine;
import com.ragulabs.djembed.core.EngineObserver;
import com.ragulabs.djembed.core.OnnxEmbeddingEngine;
import com.ragulabs.djembed.core.OnnxRerankEngine;
import com.ragulabs.djembed.core.RerankEngine;
import com.ragulabs.djembed.server.config.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * The engines being served, by model name.
 */
public final class ModelRegistry implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(ModelRegistry.class);

    private final Map<String, EmbeddingEngine> embedders;
    private final Map<String, RerankEngine> rerankers;
    private final List<String> names;

    public ModelRegistry(Map<String, EmbeddingEngine> embedders, Map<String, RerankEngine> rerankers) {
        this.embedders = Collections.unmodifiableMap(new LinkedHashMap<>(embedders));
        this.rerankers = Collections.unmodifiableMap(new LinkedHashMap<>(rerankers));
        List<String> all = new ArrayList<>(embedders.keySet());
        all.addAll(rerankers.keySet());
        this.names = List.copyOf(all);
    }

    /**
     * Loads every configured model, each reporting to the observer {@code observers} returns for its name; if one
     * fails, those already loaded are released.
     */
    public static ModelRegistry load(List<ModelConfig> models, Function<String, EngineObserver> observers) {
        Map<String, EmbeddingEngine> embedders = new LinkedHashMap<>();
        Map<String, RerankEngine> rerankers = new LinkedHashMap<>();
        try {
            for (ModelConfig model : models) {
                switch (model.task()) {
                    case EMBED -> embedders.put(model.name(),
                            OnnxEmbeddingEngine.load(model.path(), model.embeddingOptions(), observers.apply(model.name())));
                    case RERANK -> rerankers.put(model.name(),
                            OnnxRerankEngine.load(model.path(), model.rerankOptions(), observers.apply(model.name())));
                }
                log.info("Model '{}' ready ({})", model.name(), model.task());
            }
        } catch (RuntimeException e) {
            embedders.values().forEach(EmbeddingEngine::close);
            rerankers.values().forEach(RerankEngine::close);
            throw e;
        }
        return new ModelRegistry(embedders, rerankers);
    }

    /** The embedding model called {@code name}, or {@code null}. */
    public EmbeddingEngine embedder(String name) {
        return embedders.get(name);
    }

    /** The rerank model called {@code name}, or {@code null}. */
    public RerankEngine reranker(String name) {
        return rerankers.get(name);
    }

    /** The only embedding model, or {@code null} when there are none or several. */
    public String soleEmbedder() {
        return embedders.size() == 1 ? embedders.keySet().iterator().next() : null;
    }

    /** The only rerank model, or {@code null} when there are none or several. */
    public String soleReranker() {
        return rerankers.size() == 1 ? rerankers.keySet().iterator().next() : null;
    }

    /** All model names, embedding models first, in configuration order. */
    public List<String> names() {
        return names;
    }

    public Map<String, EmbeddingEngine> embedders() {
        return embedders;
    }

    public Map<String, RerankEngine> rerankers() {
        return rerankers;
    }

    public boolean contains(String name) {
        return embedders.containsKey(name) || rerankers.containsKey(name);
    }

    @Override
    public void close() {
        embedders.values().forEach(EmbeddingEngine::close);
        rerankers.values().forEach(RerankEngine::close);
    }
}

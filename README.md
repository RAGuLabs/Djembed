# Djembed

Distributed Java Embeddings: an ONNX Runtime server for embedding and reranking models, exposing Cohere-compatible
(`/v1/embed`, `/v2/embed`, `/v1/rerank`, `/v2/rerank`) and OpenAI-compatible (`/v1/embeddings`, `/v1/models`) HTTP APIs.

## Configuration

| Key | Default | Description |
|---|---|---|
| `server.host` | `0.0.0.0` | Bind address |
| `server.port` | `8080` | HTTP port |
| `server.max-request-bytes` | `33554432` | Largest accepted request body |
| `server.request-timeout-ms` | `60000` | Per-request timeout, queueing included; `0` disables it |
| `server.api-keys` | none | Bearer tokens accepted on the API routes; none leaves the API open. `/health` and `/metrics` stay open |
| `models[].name` | — | Name clients pass in the `model` request field |
| `models[].task` | — | `embed` or `rerank` |
| `models[].path` | — | Model directory, relative to the config file |
| `models[].device` | `cpu` | `cpu`, `cuda` or `cuda:N` |
| `models[].max-batch-size` | `64` | Most sequences in one forward pass |
| `models[].token-budget` | `16384` | Most `sequences × padded length` in one forward pass |
| `models[].max-input-tokens` | model limit | Tokens per sequence; lower it to bound attention memory |
| `models[].max-queued-inputs` | `8192` | Texts/documents queued across requests before answering `503` |
| `models[].long-input` | `truncate` | Embed only: `truncate` or `chunk` (word-aligned windows, averaged) |
| `models[].pooling` | from model | Embed only: `cls`, `mean` or `last_token`, for models without in-graph pooling |
| `models[].normalize` | `true` | Embed only: L2-normalise vectors |
| `models[].activation` | `sigmoid` | Rerank only: `sigmoid` or `none` (raw logit) |

| Environment variable | Description |
|---|---|
| `DJEMBED_CONFIG` | Config file, when no CLI argument is given (default `./djembed.yaml`) |
| `DJEMBED_API_KEYS` | Comma-separated API keys, added to `server.api-keys` |

See `djembed.example.yaml`. Prometheus metrics are served at `/metrics`.

## Using the core in a Java application

`djembed-core` runs the same engines in-process, without the server: tokenization, length-sorted batching across
concurrent callers, and inference. It needs Java 25 and one ONNX Runtime artifact of your choice, `onnxruntime`
(CPU) or `onnxruntime_gpu` (CUDA 12 and cuDNN 9, CPU included).

```groovy
dependencies {
    implementation 'dev.fgnm.djembed:djembed-core:0.1.0'
    runtimeOnly 'com.microsoft.onnxruntime:onnxruntime_gpu:1.29.0'
}
```

```xml
<dependency>
    <groupId>dev.fgnm.djembed</groupId>
    <artifactId>djembed-core</artifactId>
    <version>0.1.0</version>
</dependency>
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.29.0</version>
    <scope>runtime</scope>
</dependency>
```

Snapshots (`0.1.0-SNAPSHOT`) are published to `https://central.sonatype.com/repository/maven-snapshots/`.
Run the JVM with `--enable-native-access=ALL-UNNAMED`.

```java
EngineOptions gpu = EngineOptions.defaults().withDevice(Device.cuda(0));

try (EmbeddingEngine embedder = OnnxEmbeddingEngine.load(Path.of("models/bge-m3"),
             EmbeddingOptions.defaults().withEngine(gpu));
     RerankEngine reranker = OnnxRerankEngine.load(Path.of("models/bge-reranker-v2-m3"),
             RerankOptions.defaults().withEngine(gpu))) {

    Embeddings embeddings = embedder.embed(List.of("The cat sleeps on the sofa.", "Quarterly revenue grew 4%."));
    float[] first = embeddings.vector(0);

    RerankScores scores = reranker.score("Where does the cat sleep?",
            List.of("Quarterly revenue grew 4%.", "The cat sleeps on the sofa."));
    float relevance = scores.score(1);

    CompletableFuture<Embeddings> pending = embedder.embedAsync(List.of("Non-blocking call"));
}
```

Engines are thread-safe: load one per model and share it, so concurrent calls are batched together. A model folder
holds `model.onnx` (or `onnx/model.onnx`), `tokenizer.json` and `config.json`; token limit and pooling are read from
it. The options records mirror the `models[]` keys above.

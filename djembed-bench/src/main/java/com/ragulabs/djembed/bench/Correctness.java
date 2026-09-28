package com.ragulabs.djembed.bench;

import com.fasterxml.jackson.databind.JsonNode;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Checks, before any load, that the targets compute the same thing: same model, same precision, same scoring.
 * A throughput comparison between servers that disagree here would be meaningless.
 */
final class Correctness {

    record Report(Map<String, Double> minCosine, Map<String, Double> maxScoreDiff, Map<String, Boolean> sameTop,
                  Map<Workload, Double> tokensPerInput, List<String> problems) {
    }

    private Correctness() {
    }

    static Report check(HttpClient client, List<Target> targets, long seed) throws InterruptedException {
        Corpus corpus = new Corpus(seed ^ 0x5EEDL);
        List<String> texts = corpus.texts(16, 20, 200);
        String query = corpus.text(6, 12);
        List<String> documents = new ArrayList<>(corpus.texts(16, 30, 200));
        // One document repeats the query, so every correct reranker puts it first.
        documents.set(5, query + " " + documents.get(5));

        Map<String, Double> minCosine = new LinkedHashMap<>();
        Map<String, Double> maxScoreDiff = new LinkedHashMap<>();
        Map<String, Boolean> sameTop = new LinkedHashMap<>();
        Map<Workload, Double> tokensPerInput = new EnumMap<>(Workload.class);
        List<String> problems = new ArrayList<>();

        float[][] reference = null;
        float[] referenceScores = null;
        for (Target target : targets) {
            if (target.supports(Workload.QUERY)) {
                try {
                    float[][] vectors = Target.embeddings(send(client, target.embed(texts)));
                    if (reference == null) {
                        reference = vectors;
                    }
                    double min = 1;
                    for (int i = 0; i < texts.size(); i++) {
                        min = Math.min(min, cosine(reference[i], vectors[i]));
                    }
                    minCosine.put(target.name(), min);
                } catch (IOException e) {
                    problems.add(target.name() + " embed: " + e.getMessage());
                }
            }
            if (target.supports(Workload.RERANK)) {
                try {
                    float[] scores = target.scores(send(client, target.rerank(query, documents)), documents.size());
                    if (referenceScores == null) {
                        referenceScores = scores;
                    }
                    double diff = 0;
                    for (int i = 0; i < scores.length; i++) {
                        diff = Math.max(diff, Math.abs(scores[i] - referenceScores[i]));
                    }
                    maxScoreDiff.put(target.name(), diff);
                    sameTop.put(target.name(), argmax(scores) == 5);
                } catch (IOException e) {
                    problems.add(target.name() + " rerank: " + e.getMessage());
                }
            }
        }

        // Tokens per input, from the servers' own usage accounting over a sample of the actual requests.
        for (Workload workload : List.of(Workload.QUERY, Workload.INGEST)) {
            for (Target target : targets) {
                if (!target.supports(workload)) {
                    continue;
                }
                try {
                    long tokens = 0;
                    int inputs = 0;
                    for (Corpus.Payload payload : Corpus.payloads(workload, seed, 8)) {
                        tokens += send(client, target.embed(payload.texts())).path("usage").path("prompt_tokens").asLong();
                        inputs += payload.texts().size();
                    }
                    if (tokens > 0) {
                        tokensPerInput.put(workload, (double) tokens / inputs);
                        break;
                    }
                } catch (IOException e) {
                    problems.add(target.name() + " usage: " + e.getMessage());
                }
            }
        }
        return new Report(minCosine, maxScoreDiff, sameTop, tokensPerInput, problems);
    }

    private static JsonNode send(HttpClient client, HttpRequest request) throws IOException, InterruptedException {
        HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
        if (response.statusCode() != 200) {
            throw new IOException(request.uri() + " answered " + response.statusCode() + ": "
                    + new String(response.body(), StandardCharsets.UTF_8));
        }
        return Target.JSON.readTree(response.body());
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }
}

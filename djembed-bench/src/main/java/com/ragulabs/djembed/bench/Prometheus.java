package com.ragulabs.djembed.bench;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.HashMap;
import java.util.Map;

/**
 * One scrape of a Prometheus text endpoint: series keyed by {@code name{labels}}.
 */
final class Prometheus {

    private final Map<String, Double> series;

    private Prometheus(Map<String, Double> series) {
        this.series = series;
    }

    static Prometheus scrape(HttpClient client, URI uri) throws IOException, InterruptedException {
        HttpResponse<String> response = client.send(HttpRequest.newBuilder(uri).GET().build(), HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("GET " + uri + " answered " + response.statusCode());
        }
        return parse(response.body());
    }

    static Prometheus parse(String text) {
        Map<String, Double> series = new HashMap<>();
        for (String line : text.split("\n")) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            int split = line.lastIndexOf(' ');
            try {
                series.merge(line.substring(0, split).trim(), Double.parseDouble(line.substring(split + 1)), Double::sum);
            } catch (NumberFormatException e) {
                // Exemplars or timestamps this parser does not need.
            }
        }
        return new Prometheus(series);
    }

    /** Sum of every series of {@code name} whose labels contain all of {@code labels} (e.g. {@code model="bge-m3"}). */
    double sum(String name, String... labels) {
        double sum = 0;
        for (Map.Entry<String, Double> e : series.entrySet()) {
            String key = e.getKey();
            if (!key.equals(name) && !key.startsWith(name + "{")) {
                continue;
            }
            boolean matches = true;
            for (String label : labels) {
                matches &= key.contains(label);
            }
            if (matches) {
                sum += e.getValue();
            }
        }
        return sum;
    }
}

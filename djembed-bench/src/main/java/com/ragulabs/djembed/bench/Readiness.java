package com.ragulabs.djembed.bench;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;

/**
 * Waits until every target answers its health check: servers still downloading or loading models otherwise fail
 * the first requests and skew or void the run.
 */
final class Readiness {

    private static final Duration POLL = Duration.ofSeconds(2);

    private Readiness() {
    }

    /** @return false if some endpoint was still not ready after {@code timeout} */
    static boolean await(HttpClient client, List<Target> targets, Duration timeout) throws InterruptedException {
        long deadline = System.nanoTime() + timeout.toNanos();
        for (Target target : targets) {
            for (URI health : target.health()) {
                boolean announced = false;
                while (!ready(client, health)) {
                    if (System.nanoTime() > deadline) {
                        System.out.println("Not ready after " + timeout.toMinutes() + " min: " + health);
                        return false;
                    }
                    if (!announced) {
                        System.out.println("Waiting for " + health + " ...");
                        announced = true;
                    }
                    Thread.sleep(POLL);
                }
            }
        }
        return true;
    }

    private static boolean ready(HttpClient client, URI health) throws InterruptedException {
        try {
            HttpRequest request = HttpRequest.newBuilder(health).timeout(Duration.ofSeconds(5)).GET().build();
            return client.send(request, HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
        } catch (IOException e) {
            return false;
        }
    }
}

package com.ragulabs.djembed.bench;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deterministic pseudo-text: the same seed yields the same requests on every machine and for every target.
 * Mixed English and Italian vocabulary keeps the multilingual tokenizer busy the way real RAG corpora do; the
 * content itself does not matter to throughput, only the token count does.
 */
final class Corpus {

    private static final String[] WORDS = ("""
            the of and to in is that for it as with was on be by at this from or an are not have which one all their \
            there been has more when will would who so if out about up into them some could what time only other new \
            system data model search document query retrieval embedding vector index language network training result \
            performance memory server request response batch token sequence context answer question knowledge source \
            il la di che e un una per con non sono della delle degli nel nella al alla come più anche questo questa \
            legge articolo comma decreto contratto diritto obbligo termine parte giudice tribunale sentenza norma \
            codice civile penale procedura società impresa lavoro tributo imposta pagamento responsabilità danno \
            prescrizione efficacia validità disposizione regolamento ministero governo repubblica cittadino tutela""")
            .split("\\s+");

    private static final String[] PUNCTUATION = {".", ",", ";", ".", ","};

    private final SplittableRandom random;

    Corpus(long seed) {
        this.random = new SplittableRandom(seed);
    }

    String text(int minWords, int maxWords) {
        int words = minWords + random.nextInt(maxWords - minWords + 1);
        StringBuilder text = new StringBuilder(words * 8);
        boolean capitalize = true;
        for (int i = 0; i < words; i++) {
            String word = WORDS[random.nextInt(WORDS.length)];
            if (i > 0) {
                text.append(' ');
            }
            text.append(capitalize ? Character.toUpperCase(word.charAt(0)) + word.substring(1) : word);
            capitalize = false;
            if (i < words - 1 && random.nextInt(12) == 0) {
                String mark = PUNCTUATION[random.nextInt(PUNCTUATION.length)];
                text.append(mark);
                capitalize = mark.equals(".");
            }
        }
        return text.append('.').toString();
    }

    List<String> texts(int count, int minWords, int maxWords) {
        List<String> texts = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            texts.add(text(minWords, maxWords));
        }
        return texts;
    }

    /** A request's content: {@code query} is set for reranking only. */
    record Payload(String query, List<String> texts) {
    }

    /** {@code count} requests of {@code workload}, identical for a given seed. */
    static List<Payload> payloads(Workload workload, long seed, int count) {
        Corpus corpus = new Corpus(seed * 31 + workload.ordinal());
        List<Payload> payloads = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            String query = workload == Workload.RERANK ? corpus.text(6, 16) : null;
            payloads.add(new Payload(query, corpus.texts(workload.inputs, workload.minWords, workload.maxWords)));
        }
        return payloads;
    }
}

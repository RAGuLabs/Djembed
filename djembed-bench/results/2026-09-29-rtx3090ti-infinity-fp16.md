# Djembed vs Infinity, fp16

- GPU: GPU 0: NVIDIA GeForce RTX 3090 Ti
- Java 25.0.1+8-LTS, seed 42, warm-up 15 s, measured 60 s per run
- Embeddings, min cosine vs reference: djembed 1.000000
- Embeddings, min cosine vs reference: infinity 0.999958
- Rerank, max score difference vs reference: djembed 0.00e+00, planted document ranked first: true
- Rerank, max score difference vs reference: infinity 3.61e-03, planted document ranked first: true

## query (1 input(s) per request)

| target | concurrency | req/s | inputs/s | tokens/s | p50 ms | p90 ms | p99 ms | errors | GPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| djembed | 1 | 476.7 | 477 | 8104 | 2.0 | 2.4 | 5.2 | 0 | 81 |
| infinity | 1 | 80.0 | 80 | 1360 | 11.8 | 15.2 | 23.3 | 0 | 23 |
| djembed | 8 | 1706.2 | 1706 | 29005 | 4.3 | 5.7 | 9.7 | 0 | 96 |
| infinity | 8 | 379.5 | 379 | 6451 | 20.4 | 28.2 | 43.8 | 0 | 46 |
| djembed | 32 | 3428.8 | 3429 | 58290 | 8.8 | 11.3 | 17.3 | 0 | 91 |
| infinity | 32 | 625.8 | 626 | 10639 | 42.8 | 72.8 | 179.2 | 0 | 40 |
| djembed | 128 | 5197.5 | 5197 | 88357 | 24.3 | 26.3 | 32.2 | 0 | 96 |
| infinity | 128 | 881.4 | 881 | 14983 | 129.5 | 223.7 | 282.9 | 0 | 25 |

## ingest (32 input(s) per request)

| target | concurrency | req/s | inputs/s | tokens/s | p50 ms | p90 ms | p99 ms | errors | GPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| djembed | 1 | 11.8 | 377 | 101165 | 84.1 | 93.3 | 98.6 | 0 | 95 |
| infinity | 1 | 10.3 | 328 | 88126 | 98.7 | 106.1 | 112.2 | 0 | 85 |
| djembed | 8 | 12.3 | 393 | 105464 | 648.2 | 681.5 | 818.2 | 0 | 100 |
| infinity | 8 | 11.7 | 374 | 100592 | 683.5 | 704.5 | 778.2 | 0 | 98 |
| djembed | 32 | 12.5 | 401 | 107614 | 2566.1 | 3047.4 | 3121.2 | 0 | 100 |
| infinity | 32 | 11.8 | 378 | 101595 | 2713.6 | 2820.1 | 2873.3 | 0 | 95 |
| djembed | 128 | 12.5 | 399 | 107327 | 10346.5 | 11329.5 | 12255.2 | 0 | 100 |
| infinity | 128 | 11.6 | 371 | 99732 | 10952.7 | 11034.6 | 11141.1 | 0 | 93 |

## rerank (32 input(s) per request)

| target | concurrency | req/s | inputs/s | tokens/s | p50 ms | p90 ms | p99 ms | errors | GPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| djembed | 1 | 17.1 | 548 | - | 57.9 | 63.2 | 67.9 | 0 | 96 |
| infinity | 1 | 14.7 | 469 | - | 68.4 | 72.6 | 76.5 | 0 | 84 |
| djembed | 8 | 18.4 | 589 | - | 435.5 | 450.3 | 459.3 | 0 | 100 |
| infinity | 8 | 17.8 | 571 | - | 447.2 | 462.8 | 555.5 | 0 | 97 |
| djembed | 32 | 18.7 | 598 | - | 1815.6 | 1913.9 | 2150.4 | 0 | 100 |
| infinity | 32 | 17.2 | 550 | - | 1845.2 | 1950.7 | 1972.2 | 0 | 96 |
| djembed | 128 | 17.8 | 568 | - | 7024.6 | 7917.6 | 8470.5 | 0 | 100 |
| infinity | 128 | 17.3 | 554 | - | 7438.3 | 7491.6 | 7512.1 | 0 | 96 |

## Djembed relative to Infinity

Throughput above 1.00× and p99 below 1.00× favour Djembed. Runs where either server returned errors are not compared.

| workload | concurrency | throughput | p99 latency |
|---|---:|---:|---:|
| query | 1 | 5.96× | 0.22× |
| query | 8 | 4.50× | 0.22× |
| query | 32 | 5.48× | 0.10× |
| query | 128 | 5.90× | 0.11× |
| ingest | 1 | 1.15× | 0.88× |
| ingest | 8 | 1.05× | 1.05× |
| ingest | 32 | 1.06× | 1.09× |
| ingest | 128 | 1.08× | 1.10× |
| rerank | 1 | 1.17× | 0.89× |
| rerank | 8 | 1.03× | 0.83× |
| rerank | 32 | 1.09× | 1.09× |
| rerank | 128 | 1.03× | 1.13× |

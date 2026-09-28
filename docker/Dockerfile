# syntax=docker/dockerfile:1
# Build from the repository root:  docker build -f docker/Dockerfile -t djembed .

FROM eclipse-temurin:25-jdk-noble AS build
WORKDIR /src
COPY . .
# Tests run in CI; the image build only assembles the server.
RUN ./gradlew --no-daemon :djembed-server:installDist -x test

# onnxruntime_gpu 1.29 is built against CUDA 12.8 and cuDNN 9.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=eclipse-temurin:25-jre-noble /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH

# Djembed switches ONNX Runtime telemetry off in code before the runtime loads; the variable is set here too so
# it is in place before the JVM even starts.
ENV ORT_DISABLE_TELEMETRY=1 \
    DJEMBED_CONFIG=/etc/djembed/djembed.yaml

# A home directory, because the tokenizer library unpacks its native code under ~/.djl.ai.
RUN useradd --create-home --uid 10001 djembed
COPY --from=build /src/djembed-server/build/install/djembed-server /opt/djembed
USER djembed

EXPOSE 8080
# Assumes server.port 8080; loading large models on the device can take a while, hence the start period.
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/opt/djembed/bin/djembed-server"]

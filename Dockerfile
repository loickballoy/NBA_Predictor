# syntax=docker/dockerfile:1

# ---------- Étape 1 : compilation et tests du C++ ----------
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src/model
COPY model/ ./

RUN make LDFLAGS="-static-libstdc++ -static-libgcc" \
 && make test LDFLAGS="-static-libstdc++ -static-libgcc"


# ---------- Étape 2 : image d'exécution ----------
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY collector/ ./collector/
COPY pipeline/ ./pipeline/
COPY --from=builder /src/model/bin/train /src/model/bin/predict /app/bin/

ARG APP_UID=1000
RUN useradd --create-home --uid ${APP_UID} app \
 && mkdir -p /app/collector/data/raw /app/collector/data/processed /artifacts \
 && chown -R app:app /app/collector/data /artifacts
USER app

# Commit du code embarqué : il entre dans le nom de version de chaque modèle.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    PATH="/app/bin:${PATH}"

CMD ["python", "-m", "pipeline.job"]

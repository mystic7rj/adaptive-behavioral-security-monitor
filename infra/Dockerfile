# Stage 1: build environment installs dependencies once so runtime image stays minimal and reproducible.
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy dependency manifest first to maximize layer caching during iterative builds.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /app/requirements.txt

# Stage 2: runtime image copies only installed packages and application code for smaller attack surface.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/absm/.local/bin:${PATH}"

WORKDIR /app

# Create dedicated non-root runtime user to follow least-privilege container hardening.
RUN useradd --create-home --shell /bin/bash absm

# Copy only resolved dependencies from builder stage, avoiding build caches and compilers.
COPY --from=builder /install /usr/local

# Copy application source only; secrets must be injected at runtime via environment.
COPY api /app/api
COPY alerts /app/alerts
COPY config /app/config
COPY drift /app/drift
COPY enrich /app/enrich
COPY features /app/features
COPY ingest /app/ingest
COPY models /app/models
COPY monitoring /app/monitoring
COPY rl /app/rl
COPY storage /app/storage
COPY training /app/training
COPY explain /app/explain
COPY tests /app/tests

RUN chown -R absm:absm /app
USER absm

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

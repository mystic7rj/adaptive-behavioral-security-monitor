# Adaptive Behavioral Security Monitor (ABSM)

A production-grade **User and Entity Behavior Analytics (UEBA)** platform that detects insider threats, compromised credentials, and anomalous activity in real time using ensemble ML scoring, reinforcement-learning-tuned thresholds, and explainable alerting.

---

## Architecture Overview

```
                  ┌───────────┐
                  │   Kafka    │  ← Raw security events
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │  Ingest    │  Schema validation · Normalization
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │  Features  │  PII scrubbing · Rolling aggregates · Sequence building
                  └─────┬─────┘
                        │
             ┌──────────┼──────────┐
             │                     │
       ┌─────▼─────┐        ┌─────▼─────┐
       │Autoencoder │        │   LSTM    │  Reconstruction error + Sequence log-likelihood
       └─────┬─────┘        └─────┬─────┘
             │                     │
             └──────────┬──────────┘
                  ┌─────▼─────┐
                  │ Ensemble   │  Weighted score fusion
                  └─────┬─────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
    ┌─────▼────┐  ┌─────▼────┐  ┌────▼─────┐
    │  SHAP    │  │  Peer    │  │   RL     │
    │ Explainer│  │ Comparator│ │ Threshold│  Contextual bandit policy
    └─────┬────┘  └─────┬────┘  └────┬─────┘
          │             │             │
          └─────────────┼─────────────┘
                  ┌─────▼─────┐
                  │  Alerts    │  MITRE ATT&CK mapping · Severity routing · Dead-letter
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │   API      │  FastAPI · JWT/RBAC · Rate limiting · Audit logging
                  └────────────┘
```

---

## Key Capabilities

| Capability | Implementation |
|---|---|
| **Event Ingestion** | Kafka consumer with strict Pydantic schema validation (`extra="forbid"`) and field normalization |
| **Privacy-by-Design** | HMAC-based PII scrubbing applied before any feature computation or storage |
| **Behavioral Profiling** | Rolling aggregates (event counts, byte volumes, session stats) over configurable time windows from TimescaleDB |
| **Anomaly Detection** | Ensemble of a 64→8→64 **Autoencoder** (reconstruction error) and a 2-layer **LSTM** (sequence log-likelihood) |
| **Explainability** | **SHAP**-based local feature attributions so analysts understand *why* each event was flagged |
| **Peer Context** | Percentile ranking against same-role aggregate distributions for behavioral benchmarking |
| **Adaptive Thresholds** | **Contextual bandit** (ε-greedy) RL policy that tunes alert thresholds per (role, hour) context using analyst feedback |
| **Concept Drift Detection** | **ADWIN**-style sliding-window detector triggers automated retraining when score distributions shift |
| **Alert Routing** | Severity-tiered dispatch with MITRE ATT&CK tactic mapping, recommended analyst actions, and dead-letter persistence |
| **Observability** | Prometheus metrics (alert rate, FP rate, reconstruction error histogram, pipeline lag, feature drift scores) |
| **SLA Monitoring** | Ingest-to-alert latency tracking with configurable breach thresholds |
| **API Security** | JWT validation with issuer checks, role-based access control, rate limiting (slowapi), CORS, security headers, and structured audit logging |

---

## Project Structure

```
├── ingest/                 # Kafka consumer, event schema validation, field normalization
├── features/               # PII scrubber, profile builder, rolling aggregates, sequence builder
├── models/                 # Autoencoder, LSTM, ensemble scorer, model registry, online updater
├── training/               # Autoencoder training loop
├── drift/                  # ADWIN-style concept drift detector, retraining scheduler
├── rl/                     # Contextual bandit policy, reward calculator, feedback collector, policy guard
├── explain/                # SHAP-based explainability for autoencoder anomaly scores
├── enrich/                 # Peer-group percentile comparison using role-aggregated statistics
├── alerts/                 # Alert builder (MITRE mapping + severity), dispatcher with dead-letter
├── monitoring/             # Prometheus metrics, model health (p95 tracking), SLA checker, data drift
├── api/                    # FastAPI app, JWT/RBAC auth, rate limiting, audit middleware, routes
├── storage/                # TimescaleDB batch writer
├── config/                 # Pydantic-based settings with env/dotfile loading (extra="forbid")
├── infra/                  # Multi-stage Dockerfile (non-root), docker-compose (Kafka, TimescaleDB, Redis, Prometheus, Grafana)
├── tests/                  # Unit tests (ingest, features, models, drift, RL, alerts, API, monitoring)
│   └── e2e/                # End-to-end synthetic threat scenarios (insider threat, compromised credentials)
└── .github/workflows/      # CI pipeline: ruff, mypy --strict, pytest, bandit, safety, Docker build + push
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 · Strict typing (`mypy --strict`) |
| ML Framework | PyTorch (autoencoder, LSTM) · SHAP (explainability) · SciPy |
| API | FastAPI · Uvicorn · Starlette |
| Auth | python-jose (JWT HS256) · Role-based access control |
| Streaming | Confluent Kafka |
| Database | TimescaleDB (PostgreSQL) · asyncpg |
| Cache/State | Redis (TLS) · msgpack serialization |
| Observability | Prometheus · Grafana |
| Infrastructure | Docker (multi-stage, non-root) · Docker Compose |
| CI/CD | GitHub Actions → GHCR |
| Code Quality | Ruff · mypy --strict · Bandit (SAST) · Safety (CVE scanning) |

---

## Getting Started

### Prerequisites

- Python ≥ 3.11
- Docker & Docker Compose (for infrastructure services)

### 1. Clone and install

```bash
git clone https://github.com/mystic7rj/adaptive-behavioral-security-monitor.git
cd adaptive-behavioral-security-monitor
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Kafka, TimescaleDB, Redis, and JWT credentials
```

| Variable | Description |
|---|---|
| `KAFKA_BROKER` | Kafka bootstrap server(s) |
| `KAFKA_TOPIC` | Event topic to consume |
| `KAFKA_GROUP_ID` | Consumer group ID |
| `DB_URL` | TimescaleDB connection string |
| `SCRUBBER_HMAC_KEY` | HMAC key for PII scrubbing |
| `REDIS_URL` | Redis connection URL (TLS supported) |
| `MODEL_REGISTRY_DIR` | Path for serialized model storage |
| `JWT_SECRET_KEY` | Secret for JWT signature validation |
| `JWT_ISSUER` | Expected JWT issuer claim |
| `LOG_LEVEL` | Logging level (default: `INFO`) |

### 3. Start infrastructure

```bash
docker compose -f infra/docker-compose.yml up -d
```

This brings up Kafka, TimescaleDB, Redis, Prometheus, and Grafana on an isolated bridge network.

### 4. Initialize the database

```sql
CREATE TABLE IF NOT EXISTS security_events (
    user_id       UUID NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL,
    event_type    TEXT NOT NULL,
    ip            TEXT,
    email         TEXT,
    session_duration_s DOUBLE PRECISION,
    bytes_out     BIGINT
);

SELECT create_hypertable('security_events', 'timestamp');
```

### 5. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6. Start the Kafka consumer

```bash
python -c "from ingest.kafka_consumer import consume_loop; consume_loop(lambda e: print(e))"
```

---

## Testing

### Unit tests

```bash
pytest tests/ --ignore=tests/e2e -q
```

### End-to-end scenario tests

```bash
python -c "from tests.e2e.test_insider_threat import run_scenario; assert run_scenario()"
python -c "from tests.e2e.test_compromised_creds import run_scenario; assert run_scenario()"
```

### Code quality gates (same as CI)

```bash
ruff check .                         # Linting
mypy --strict .                      # Type checking
bandit -r alerts api config drift enrich features ingest models monitoring rl storage training explain   # SAST
safety check --full-report           # Dependency CVE scan
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) enforces two gate types:

**Pull Request → `pr-quality-gates`**
- Ruff lint check
- mypy strict type checking
- Unit test suite (e2e excluded)
- Bandit SAST scan across all source packages
- Safety dependency vulnerability scan

**Push to `main` → `main-release`**
- Full dependency install
- Docker image build (multi-stage, non-root)
- E2E synthetic threat scenario validation
- Image publish to GitHub Container Registry (GHCR)

---

## ML Pipeline Details

### Anomaly Detection (Ensemble)

The ensemble scorer fuses two complementary signals:

- **BehaviorAutoencoder** — A 64→32→8→32→64 feedforward autoencoder trained on normal behavior vectors. High reconstruction error (MSE) indicates behavioral anomaly.
- **SessionLSTM** — A 2-layer LSTM with learned token embeddings that models event sequences. Low sequence log-likelihood flags unusual action ordering.

Scores are combined via configurable weighted average (default 50/50).

### Adaptive Thresholds (RL)

A **contextual bandit** with ε-greedy exploration (ε=0.05) selects alert thresholds from a discrete action set `{0.3, 0.4, 0.5, 0.6, 0.7, 0.8}` per `(user_role, hour_bucket)` context.

- **Rewards** are asymmetric: TP=+1.0, FP=-0.5, FN=-2.0, critical FN=-4.0
- Q-values are persisted to disk and survive restarts
- A **policy guard** constrains threshold drift to prevent unsafe exploration

### Drift Detection

An ADWIN-style sliding-window detector monitors score distributions. When a statistically significant mean shift is detected between window halves, retraining is scheduled automatically.

---

## Security Posture

- **Strict schema validation** — Pydantic models with `extra="forbid"` reject unexpected fields
- **PII scrubbing** — HMAC-based pseudonymization before any feature computation
- **JWT authentication** — HS256 signature + issuer validation, no fallback on failure
- **Role-based access control** — Dependency-injected role enforcement at route level
- **Rate limiting** — slowapi-based request throttling
- **Security headers** — `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`
- **Audit logging** — Middleware records all API requests with structured context
- **Non-root containers** — Dedicated `absm` user in Docker runtime stage
- **Network isolation** — Docker Compose services on private bridge; only API (8000) and Grafana (3000) are externally exposed
- **SAST + CVE scanning** — Bandit and Safety gates in CI

---

## License

This project is proprietary. All rights reserved.

# Adaptive Behavioral Security Monitor (ABSM) — Ingestion

This project provides a secure ingestion pipeline that reads events from Kafka, normalizes them, validates against strict schemas, and writes batches to TimescaleDB.

## Structure
```
absm/
  ingest/          Kafka consumer, schema validation, normalization
  storage/         TimescaleDB batch writer
  config/          Environment-based settings
  tests/           Pytest coverage for ingest and normalization
```

## Setup
1. Copy `.env.example` to `.env` and fill in values.
2. Install dependencies:
```
pip install -e .
```

## Run tests (Week 1 — Testing Commands)
```
pytest -q
```

## Run the Kafka consumer
```
python -c "from ingest.kafka_consumer import consume_loop; consume_loop(lambda e: print(e))"
```

## TimescaleDB table
Create a table compatible with the writer:
```
CREATE TABLE IF NOT EXISTS security_events (
  user_id UUID NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  event_type TEXT NOT NULL
);
```

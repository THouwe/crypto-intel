#!/usr/bin/env sh
# Deployment entrypoint: one-time initial DB populate, then run the service.
#
# Use as the Railway start command for the QUERY API service, e.g.:
#   sh deploy/start.sh uvicorn crypto_intel.web.app:app --host 0.0.0.0 --port "$PORT"
#
# The initial ingest runs ONCE — `--if-empty` skips it as soon as the store has
# content, so restarts don't re-ingest. Disable it with INITIAL_INGEST=0.
#
# The recurring 3h refresh is a SEPARATE Railway cron service, not this script:
#   crypto-intel ingest --sources news,exchange,regulator --lookback-hours 3
#
# Requires: STORE_BACKEND=pgvector and DATABASE_URL in the environment, and the
# schema provisioned (deploy/schema.sql) — the app also self-creates it.
set -eu

if [ "${INITIAL_INGEST:-1}" = "1" ]; then
  echo "[start] initial populate (once, --if-empty) ..."
  # 168h = 7 days, matching the retention window (deploy/retention.sql).
  crypto-intel ingest --sources news,exchange,regulator --lookback-hours 168 --if-empty \
    || echo "[start] initial ingest failed; starting service anyway"
fi

echo "[start] launching: $*"
exec "$@"

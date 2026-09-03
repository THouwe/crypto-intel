-- Rolling-window retention for the pgvector store, run inside Postgres via
-- pg_cron. This is the DB-native alternative to running `crypto-intel prune`
-- from the app/cron. Use one or the other, not both.
--
-- Safe with the recurring ingest: ingest de-duplicates against the `chunks`
-- table (not the JSONL archive), so documents deleted here are re-embedded on
-- the next ingest run and the store re-populates. See docs/DEPLOYMENT.md §5.
--
-- Target: Supabase (pg_cron is available; enable it in Dashboard → Database →
-- Extensions, or with the CREATE EXTENSION below). Keeps only the last 7 days.

-- 1) Enable pg_cron (no-op if already enabled).
create extension if not exists pg_cron;

-- 2) Schedule an hourly delete of chunks older than 7 days (runs at :05).
--    cron.schedule(job_name, cron_expr, sql) — re-running with the same job_name
--    updates the existing schedule.
select cron.schedule(
  'crypto_intel_retention',
  '5 * * * *',
  $$ delete from chunks where published_at < now() - interval '7 days' $$
);

-- --- Management ------------------------------------------------------------
-- Inspect scheduled jobs:
--   select jobid, schedule, command, active from cron.job;
-- Inspect recent runs:
--   select * from cron.job_run_details order by start_time desc limit 20;
-- Change the window: re-run cron.schedule(...) above with a new interval.
-- Remove the job:
--   select cron.unschedule('crypto_intel_retention');
--
-- Note: on Supabase, pg_cron runs in the `postgres` database. If your app
-- connects to a different database, schedule from there or use cron.schedule_in_database().

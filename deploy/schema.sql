-- Crypto Market Intelligence — pgvector schema for Supabase / Postgres.
--
-- Run once in the Supabase SQL Editor (Dashboard -> SQL Editor -> New query),
-- or via psql. The app also creates this automatically on first use
-- (PgVectorStore.init_schema), but provisioning it here up front is cleaner for
-- a fresh deploy. Embedding dimension is 384 (all-MiniLM-L6-v2).

create extension if not exists vector;

create table if not exists chunks (
    id            text primary key,
    doc_id        text not null,
    chunk_index   integer not null,
    source        text not null,
    source_name   text not null,
    url           text not null,
    published_at  timestamptz not null,
    assets        text[] not null default '{}',
    text          text not null,
    embedding     vector(384) not null
);

-- Filter indexes used by retrieval (time window + asset membership).
create index if not exists chunks_published_at_idx on chunks (published_at);
create index if not exists chunks_assets_idx on chunks using gin (assets);

-- Optional approximate-NN index for cosine search. Not needed at a few thousand
-- rows (exact scan is fast); enable it if the corpus grows large.
-- create index if not exists chunks_embedding_hnsw
--   on chunks using hnsw (embedding vector_cosine_ops);

-- Retention (rolling window) is scheduled separately — see deploy/retention.sql.

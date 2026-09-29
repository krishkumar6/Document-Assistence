# Document Assistant

Ask questions over a folder of PDFs and get answers with **file + page citations**.
PDFs are chunked and embedded into **Chroma** locally; `/ask` retrieves the top
chunks and has an LLM answer with citations mapped back to the exact page. The
LLM can be free (Groq free tier, Gemini free tier, or local Ollama) or Claude.
Two chunk sizes are indexed side by side and compared on a labelled question set.

```
PDF ──pypdf──► pages ──chunker──► chunks (+ per-page char spans) ──MiniLM──► Chroma
                                                                  docs_small / docs_large
question ──► top-k chunks ──► Claude (1 document per chunk, 1 text block per page)
         ──► content_block_location citations ──► {file, page, quote}
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements-dev.txt  # app + tests/eval tools (the Docker image uses requirements.txt)

python scripts/make_sample_pdfs.py   # optional: 3 sample PDFs with known answers
python -m app.ingest                 # ingest data/pdfs into both collections

copy .env.example .env               # then paste your free Groq key into .env
uvicorn app.main:app --reload
```

Then open **http://localhost:8000/**. That's the page for everyday users: drop in
PDFs, type a question, and pick **Quick answer** (`/ask`) or **Research** (the
agent). Answers show numbered sources you can click. `/docs` is the developer API
reference.

The first ingest downloads the local embedding model (~80 MB, `all-MiniLM-L6-v2`).
Only `/ask` needs an LLM; ingestion and retrieval run fully offline.

## Deploy to Render (free)

`render.yaml` is a Render Blueprint set up for the **free plan**. It was tested locally
under the free plan's exact limits (512 MB RAM, 0.1 CPU, disk wiped on restart).

**Steps:**
1. Push this repo to GitHub (private is fine).
2. In Render: **New → Blueprint**, connect GitHub, pick the repo. Render reads `render.yaml`.
3. When it asks for the two secret values, enter:
   - `GROQ_API_KEY`: your key from console.groq.com
   - `APP_PASSWORD`: the password your users will type
4. **Apply.** The first build takes ~10 minutes: it installs everything and builds the models
   and your library's index into the image.
5. Open `https://document-assistant-xxxx.onrender.com` and log in with any username and
   your `APP_PASSWORD`.

**How the free plan behaves (measured locally under the same limits):**

| | |
|---|---|
| Memory | 144 MB of 512 MB (reranker off; hybrid vector + BM25 search still on) |
| Wake from sleep | Render sleeps the app after 15 min idle; waking takes ~1 min, then ~40 s to start |
| First question after waking | ~20–25 s (models finish loading) |
| Later questions | ~4–7 s |
| Accuracy | right page first 88% of the time on our eval (96% with the reranker, which needs a paid plan) |

**Your documents on the free plan.** Render wipes the disk whenever the app restarts or sleeps, so:
- **PDFs in the `library/` folder are permanent.** Put the documents you want always available
  there, commit and push; Render rebuilds automatically. They're indexed at build time and
  restored on every start in seconds.
- **PDFs uploaded on the web page are temporary.** They're gone after the next restart.
  Good for trying a file out, not for keeping it.
- The repo ships with the three sample PDFs in `library/`. Replace them with your own.

**Upgrading later** (faster, keeps uploads, reranker back on): in `render.yaml` set
`plan: 1c-2g`, add the disk block shown in its comments, set `RETRIEVAL_RERANK: "true"`, and push.
The app needs ~0.6 GB at peak with the reranker, more than Render's 512 MB plans.

**Safety defaults when deployed:**
- **Password:** the app **refuses to start** without `APP_PASSWORD`, unless you set
  `ALLOW_PUBLIC=true` on purpose. It protects the page and API; only `/health` stays open for
  Render's health check.
- **Upload limits:** 10 files per upload, 10 MB (free plan) and 500 pages per file. Files must
  be real PDFs, and a rejected batch saves nothing.
- **Rate limit:** 20 questions per minute per visitor, to protect your free Groq quota.
- **Secrets:** your Groq key lives only in Render's environment settings. `.env` is git-ignored
  and excluded from the image.
- **Tuning for a small CPU:** `ORT_THREADS=1` runs the models on one thread. On 0.1 CPU this made
  searches ~4x faster (12 s → 3 s), because the default one thread per visible core made the
  threads fight over the tiny CPU share.

The same `Dockerfile` runs on any Docker host (`docker run -p 8000:8000 -e APP_PASSWORD=... -e GROQ_API_KEY=... -v data:/data`).

## Choosing the LLM (free options)

Set `LLM_PROVIDER` in `.env`:

| `LLM_PROVIDER` | Cost | Setup | Default model |
|---|---|---|---|
| `groq` | Free tier: 30 req/min, 1K req/day, 8K tokens/min | Key from https://console.groq.com/keys → `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| `gemini` | Free tier | Key from https://aistudio.google.com/apikey → `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `ollama` | Free, runs locally, no key | Install [Ollama](https://ollama.com), `ollama pull qwen2.5:3b` | `qwen2.5:3b` (fits 8 GB RAM) |
| `anthropic` | Paid (no free API tier) | `ANTHROPIC_API_KEY` | `claude-opus-5` |

If `LLM_PROVIDER` is unset, it uses `anthropic` when `ANTHROPIC_API_KEY` is set and
`ollama` otherwise. `LLM_MODEL` / `LLM_BASE_URL` override the model and endpoint, so
any other OpenAI-compatible server works too. `GET /health` shows the active provider.

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/ingest` | Multipart upload of one or more PDFs (`files`). Saved to `data/pdfs` and indexed; re-uploading a file replaces its old chunks. In `/docs`: Try it out → Choose File → Execute. |
| `POST` | `/ingest/folder` | Re-index every PDF already in `data/pdfs`. |
| `POST` | `/ask` | `{"question": "...", "chunk_config": "small" \| "large", "top_k": 5}` |
| `POST` | `/ask/compare` | Same question answered with both chunk configs, side by side. |
| `POST` | `/ask/stream` | `/ask` as a Server-Sent Events stream: text arrives as it's written. See [Streaming](#streaming). |
| `POST` | `/agent` | Multi-step agent with tools: `{"question": "...", "max_steps": 6, "allow_web": true}`. See [Agent](#agent). |
| `POST` | `/agent/stream` | The agent as a stream: every step, tool call, tool result and answer word, live. |
| `GET` | `/stats` | Chunk and document counts per collection, plus answer-cache hits/misses. |
| `DELETE` | `/cache` | Empty the answer cache. |
| `GET` | `/health` | Liveness. |

```bash
curl -X POST localhost:8000/ingest -F "files=@report.pdf"
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "What is the minimum password length?"}'
```

Response shape (values illustrative):

```json
{
  "answer": "Passwords must be at least 14 characters long[1].",
  "citations": [
    {"id": 1, "file": "helios_security_policy.pdf", "pages": [2],
     "quotes": ["Passwords must be at least 14 characters long and must not appear in the breached-password list checked at login."]}
  ],
  "retrieved": [{"file": "helios_security_policy.pdf", "page_start": 2, "page_end": 2, "distance": 0.31, "chars": 486}],
  "chunk_config": {"name": "small", "size": 500, "overlap": 100},
  "model": "claude-opus-5", "input_tokens": 1420, "output_tokens": 38
}
```

Interactive docs: http://localhost:8000/docs

## Retrieval: hybrid search + reranker

```
question ─┬─ vector search (MiniLM embeddings, Chroma) ──┐
          │                                             ├─ RRF fusion ─ top 12 ─ cross-encoder reranker ─ top k
          └─ BM25 keyword search (in-memory index) ─────┘
```

- **Vector search** finds passages with similar *meaning* ("how long are logs kept" ≈ "retained for 18 months").
- **BM25** finds exact *terms* that embeddings blur: codes (`NW-IR-3`), numbers (`3.85`), names.
  The tokenizer keeps `AES-256` whole and also indexes `aes` and `256`.
- **Reciprocal Rank Fusion** merges the two ranked lists by rank (`1/(60+rank)`), so their different score scales never need calibrating.
- **Reranker:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`, ONNX via `fastembed`, ~80 MB, runs locally, free)
  reads the question and each candidate *together* and re-scores them. That's more accurate than
  either retriever, but slower, so it only runs on the fused top 12.

Defaults: `hybrid` + rerank. Override per request (`"search_mode": "vector" | "bm25" | "hybrid"`,
`"rerank": false`) or globally (`RETRIEVAL_MODE`, `RETRIEVAL_RERANK`, `RERANK_MODEL`, `CANDIDATE_POOL`).
Every entry in `retrieved` shows how it was ranked, e.g.
`{"vector_rank": 1, "bm25_rank": 2, "rrf": 0.0325, "rerank": 6.07}`.

**Measured** (`python -m eval.compare_retrieval`, free, no LLM calls). There are 26 questions: the
original 16, plus 5 *exact-term* (codes and numbers) and 5 *paraphrase* (reworded, few shared words).
"hit@1" means the right page was ranked first.

| strategy | small: hit@1 | exact-term | paraphrase | large: hit@1 | ms/query (small) |
|---|---|---|---|---|---|
| vector | 88% | 80% | 80% | 77% | 208 |
| bm25 | 85% | **100%** | 60% | 88% | 1 |
| hybrid | 88% | 80% | 80% | 88% | 204 |
| vector + rerank | **96%** | **100%** | **100%** | **100%** | 620 |
| **hybrid + rerank** (default) | **96%** | **100%** | **100%** | **100%** | 613 |

What it shows:
- **The reranker is the big win**: 88→96% (small chunks) and 77→100% (large), and it fixes
  every paraphrase miss. The cost is ~0.4s per search on a laptop CPU; a 12-candidate pool
  matched 20 candidates' accuracy at ~35% less time.
- **BM25 and vectors fail on different questions.** BM25 alone gets every exact-term question
  but only 60% of paraphrases; vectors are the other way round.
- **Hybrid + rerank = vector + rerank on this corpus**, because the corpus is tiny (44 chunks):
  vector search already had the right page in its top 5 for all 26 questions, so there was
  nothing for BM25 to rescue. Hybrid is kept on because it costs ~nothing and matters on
  larger libraries, where an exact code can fall outside vector search's top candidates.
  That benefit is expected but **not demonstrated here**; re-run the eval on your own
  documents to check.

The BM25 index is built in memory from the collection on first search and rebuilt
automatically after an ingest. The reranker loads in the background at server start.

## Agent

`/ask` does one retrieval and one answer. `/agent` is for questions that need
several lookups, arithmetic, or outside information. The model decides which
tools to call, sees the results, and repeats until it can answer.

| Tool | What it does |
|---|---|
| `search_docs(query, max_results=4)` | Searches your ingested PDFs; returns passages labelled `[S#]` with file + page. |
| `calculator(expression)` | Exact arithmetic via a safe AST evaluator (no `eval`): `+ - * / // % ** ^`, `sqrt`, `log`, `round`, `min`, `max`, `pi`... |
| `web_search(query, max_results=5)` | DuckDuckGo (free, no key). Returns titles, URLs and snippets labelled `[S#]`. Off with `"allow_web": false`. |

```bash
python -m app.agent "An Orbit Cafe grosses $300,000. What do royalty plus marketing fund cost per year?"
```
```
[a0374b92ee1c] step 1  search_docs({"query": "Orbit Cafe royalty marketing fund ..."}) -> [S1] (orbit_cafe_franchise_handbook.pdf, page 1) ... (424ms)
[a0374b92ee1c] step 2  calculator({"expression": "0.08 * 300000"}) -> 24000 (0ms)
[a0374b92ee1c] step 3  final answer (762ms)
... 6% royalty plus 2% marketing fund = 8% of gross sales[1] ... $24,000 in total.
  [1] orbit_cafe_franchise_handbook.pdf p.1
stopped: answered | steps: 3/6
```

**Citations** from both tools share one `[S#]` label space and come back in
`citations` as `{"kind": "doc", "file", "pages"}` or `{"kind": "web", "url", "title"}`,
so one answer can cite a PDF page and a web page side by side.

**Step log.** A step is one model turn. Each one is logged with the model's text,
every tool call (input, output preview, error flag, duration), and any recovery
note. The log is:
- returned in the response as `steps`,
- printed live in the server console (`AGENT [run_id] step 2 calculator(...) -> ...`),
- appended with the full run to `logs/agent_runs.jsonl` (one JSON line per run).

**Step limit.** `max_steps` (default 6, max 12) caps model turns. On the last
allowed turn, tools are switched off and the model is told to answer with what it
has and say what it couldn't verify; `stopped_reason` is then `"step_limit"`
(otherwise `"answered"`). Example: a two-part question with `max_steps: 2`
answered the franchise-cost part with citations and said it could not verify the
retention-period part. With `max_steps: 8` it made 3 searches and 2 calculator
calls and answered both parts.

**Failure handling.**
- **Tool errors** (bad arguments, unknown tool, invalid JSON, division by zero,
  network failure) go back to the model as error results so it can correct itself.
  They never crash the run.
- **Malformed tool calls:** if Groq rejects a turn as a malformed tool call
  (gpt-oss sometimes reaches for a browser tool that doesn't exist), the turn is
  retried with some sampling randomness, then answered without tools if needed.
- **Untrusted content:** tool output is framed as data, and the system prompt
  tells the model to ignore instructions found inside documents or web pages.

**Cost on free tiers.** A multi-step run resends the growing conversation each
turn: a 6-step run used ~10K input tokens. That exceeds Groq's free 8K
tokens/min, so the client waits and retries (a run may pause ~20s). Keep
`max_steps` low, or set `"allow_web": false`, to stay fast.

## Streaming

`/ask/stream` and `/agent/stream` return `text/event-stream`: one `data: {json}`
line per event. The web page uses these, so answers appear word by word and
research steps appear as they happen.

| Event | When | Fields |
|---|---|---|
| `retrieved` | `/ask/stream`: passages found | `passages` [{file, page_start, page_end}] |
| `start` | agent run begins | `run_id`, `max_steps` |
| `step_start` | a model turn begins | `step`, `forced` (last step, tools off) |
| `delta` | model writes text | `text` (raw: source labels like `[S2]` not yet resolved) |
| `retry` | a malformed turn is being retried | `step`, `note` (discard that step's streamed text) |
| `tool_call` / `tool_result` | agent uses a tool | `tool`, `input` / `output_preview`, `is_error`, `ms` |
| `step_end` | a model turn ends | `log` (the step's log entry incl. tokens) |
| `done` | finished | same body as `/ask` or `/agent` (resolved `[n]` markers, citations, usage) |
| `error` | failure after streaming started | `status`, `message` |

```bash
curl -N -X POST localhost:8000/ask/stream -H "content-type: application/json" -d '{"question": "What are the data classification levels?"}'
```

Measured (Groq): a research run's first tool call reaches the browser at ~1s
instead of the whole answer at the end. Quick answers on Groq finish in under a
second either way, so streaming matters most for the agent and for slower models
like local Ollama. Responses send `X-Accel-Buffering: no` so a reverse proxy (e.g.
nginx) passes the stream through when deployed.

## Caching

Two separate layers:

**1. Provider prompt caching:** the model provider reuses work for a prompt that
starts with the same text as a recent one. Cached tokens are cheaper and, on
Groq, don't count toward the free-tier token limit. It only works if prompts
start identically, so the code is laid out for it:
- Fixed instructions and the tool list come first; they never change between requests.
- The agent's conversation is **append-only**: each turn's request begins with the
  previous request's exact messages (a test checks this). That makes every agent
  turn after the first a cache candidate.
- **Groq and Ollama:** automatic. Groq reported 768 of 12,509 input tokens as cached
  on a 6-step agent run; short prompts showed no cache hits in testing.
- **Claude:** the agent loop sets `cache_control`, so each turn reads earlier turns
  from cache. `/ask` deliberately doesn't: each prompt holds different passages,
  so a cache write (billed at 1.25× input) would rarely be read back.
- Every response reports `input_tokens` and `cached_input_tokens` (per step for the
  agent, and in `logs/agent_runs.jsonl`); the web page shows them under each answer.

**2. Answer cache (`/ask` only):** the same question (ignoring case and spacing),
with the same retrieved passages and the same model, returns the saved answer
with no LLM call: 0.3s and 0 tokens instead of ~3s. The key includes the passage
text, so adding or changing PDFs makes old entries stop matching on their own.
It's in memory (cleared on restart), keeps 256 answers for 1 hour by default
(`ANSWER_CACHE_SIZE`, `ANSWER_CACHE_TTL`; 0 disables), and skips truncated or
refused answers. The agent isn't answer-cached, because web results change.

## How citations stay page-accurate

Chunks are allowed to cross page boundaries (so a sentence split over two pages
isn't cut in half), but each chunk stores the character ranges that came from
each page. At answer time a chunk is split back into **one piece per page**, so
every citation resolves to a single file + page even when the chunk spans pages 3–4.

- **Free / OpenAI-compatible models:** each page piece is listed in the prompt as
  `[S1] (file.pdf, page 3)`, and the model is told to cite those labels. The code
  then maps the labels to file + page and renumbers them `[1]`, `[2]`…
  Labels that don't exist (e.g. a made-up `[S9]`) are dropped, so a citation always
  points at text that was really retrieved. `quotes` holds the cited excerpt.
- **Claude:** each chunk is a `document` with one text block per page piece;
  Claude's native `content_block_location` citations point at block indices and
  `quotes` holds the exact span Claude quoted.

Only sources the model actually cited are returned as citations; everything that
was retrieved is listed under `retrieved`. If the excerpts don't contain the
answer, the model is told to say so instead of answering from general knowledge.

## Chunk size comparison

| | **small** | **large** |
|---|---|---|
| chunk size / overlap (chars) | 500 / 100 | 2000 / 200 |

Run it yourself:

```bash
python -m eval.compare_chunks -k 3          # retrieval only, free
python -m eval.compare_chunks -k 3 --llm    # + answer & citation accuracy (2 x 16 LLM calls; free on Groq/Ollama)
```

`eval/questions.json` holds labelled questions, each with the file and page that
contains the answer. (The chunk-size results below were measured on the first 16,
with vector search only, before the retrieval upgrade.) The sample corpus is 3 fictional PDFs × 4 pages. It
deliberately repeats values across documents ("30 days", "4 hours", "quarterly")
so retrieval has near-duplicate distractors to get past.

**Retrieval results (k = 3, measured on this corpus):**

| metric | small (500) | large (2000) |
|---|---|---|
| chunks in index | 35 | 7 |
| hit@1 (gold page ranked first) | **93.8%** | 87.5% |
| hit@3 | 100% | 100% |
| MRR | **0.969** | 0.938 |
| context precision (share of retrieved text from the gold page) | **0.69** | 0.56 |
| avg context sent to Claude | **1,295 chars** | 5,213 chars |

At k = 5 the gap in context size widens: 2,148 vs 8,798 chars, precision 0.58 vs 0.33.

**End-to-end results (k = 3, Groq free tier, `openai/gpt-oss-120b`, 2 × 16 calls):**

| metric | small (500) | large (2000) |
|---|---|---|
| answer accuracy (expected fact in the answer) | 100% | 100% |
| citation accuracy (a citation points at the gold file + page) | 100% | 100% |
| avg input tokens per question | **494** | 1,341 |

On single-fact questions, both sizes produce correct, correctly cited answers. The
difference is cost and headroom: small chunks use **2.7× fewer input tokens**. That
matters on a free tier capped at 8K tokens/min, where small fits ~16 questions a
minute and large ~6.

**What this says**

- **Small chunks rank the right page first more often and send ~4× less text to
  the model.** Less text means lower cost per `/ask`, and a strong model still
  answers correctly from the smaller context.
- **Large chunks recover by recall.** Each one covers ~2 pages, so anything
  retrieved at all tends to include the answer. Hit@3 reaches 100% for both, but
  for large chunks that means sending about 45% of the whole corpus on every
  question. That doesn't scale.
- **Large chunks lose to truncation and dilution.** MiniLM only reads the first
  256 word pieces (~1,000 chars) of each chunk, so a 2,000-char chunk is embedded
  from its first half. For "how quickly must access be revoked", the answer sits
  at char 1,799 of the best-matching chunk, outside what was embedded. The
  overlapping neighbour chunk has it at char 29 but still ranked second (cosine
  distance 0.630 vs 0.624), because that chunk mixes three pages of unrelated
  policy. If you want large chunks, pair them with a long-context embedding model.
- **Where large chunks win:** answers that depend on surrounding context (a
  procedure spread over several paragraphs, or "compare X and Y" within one
  section). The sample questions are single-fact lookups, which favour small chunks.

**Caveat:** 12 pages is a small corpus, so hit@k saturates. Treat these numbers as
directional and re-run `eval/compare_chunks.py` on your own PDFs with your own
labelled questions before choosing. `small` is the default (`DEFAULT_CHUNK_CONFIG`).

## Project layout

```
app/
  config.py      chunk configs, paths, model (env-overridable)
  pdf_loader.py  pypdf page extraction
  chunking.py    page-aware chunker with per-page spans
  store.py       Chroma collections (one per chunk config)
  retrieval.py   vector + BM25 hybrid search, RRF fusion, cross-encoder reranking
  ingest.py      ingest pipeline + CLI
  answer.py      LLM call (Claude or OpenAI-compatible) -> answer + {file, pages, quotes}
  agent.py       tool-using agent loop: step log, step limit, Claude + OpenAI-compatible backends
  tools.py       search_docs, calculator, web_search + source labelling
  answer_cache.py  local /ask answer cache (TTL + LRU)
  security.py    password (HTTP Basic), per-IP rate limit, refuse-to-start-public guard
  seed.py        permanent library: index library/ at build, restore into an empty data folder at start
library/         PDFs that are always available when deployed (built into the image)
  static/        the web page served at /
  main.py        FastAPI app
eval/            labelled questions + comparison harness
scripts/         sample PDF generator
tests/           chunker, citation-mapping, grading and agent-loop tests (no API calls)
logs/            agent_runs.jsonl (created on first agent run, git-ignored)
```

## Configuration

All of these can go in `.env` (see `.env.example`).

| Env var | Default |
|---|---|
| `LLM_PROVIDER` | `anthropic` if `ANTHROPIC_API_KEY` is set, else `ollama` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | — (the one for your provider) |
| `LLM_MODEL` | per provider (table above) |
| `LLM_BASE_URL` | per provider |
| `DOCS_DIR` | `data/pdfs` |
| `CHROMA_DIR` | `chroma_db` |
| `DEFAULT_CHUNK_CONFIG` | `small` |
| `RETRIEVAL_MODE` / `RETRIEVAL_RERANK` | `hybrid` / `true` |
| `RERANK_MODEL` / `CANDIDATE_POOL` | `Xenova/ms-marco-MiniLM-L-6-v2` / `12` |
| `MODEL_CACHE_DIR` | `models` (reranker download, git-ignored) |
| `ANSWER_CACHE_TTL` / `ANSWER_CACHE_SIZE` | `3600` seconds / `256` answers (0 disables) |
| `LOG_DIR` | `logs` |
| `DATA_DIR` | unset locally; `/data` in Docker. Holds `pdfs/`, `chroma/`, `logs/` (Railway's `RAILWAY_VOLUME_MOUNT_PATH` also works) |
| `APP_PASSWORD` / `ALLOW_PUBLIC` | unset / `false`. On a hosting platform, a password is required unless `ALLOW_PUBLIC=true` |
| `MAX_UPLOAD_MB` / `MAX_FILES_PER_UPLOAD` / `MAX_PDF_PAGES` | `20` / `10` / `500` |
| `RATE_LIMIT_PER_MINUTE` | `20` questions per client IP (0 disables) |
| `ORT_THREADS` | `0` (ONNX default). Set `1` on tiny CPU shares like Render free |
| `SEED_DIR` | `seed` locally, `/app/seed` in Docker: prebuilt index of `library/`, copied into an empty data folder at start |

Free tiers are rate-limited: 429s are retried with backoff, and the eval waits
out the per-minute window. With Claude, `/ask` enables the API's server-side
refusal fallback (`fallbacks: "default"`), so if the primary model declines,
another model answers within the same request.

## Swapping Chroma for pgvector

All vector-store access goes through `app/store.py` (`replace_document`, `search`,
`stats`). To use Postgres, reimplement those three functions against a table
like `chunks(id text primary key, config text, source text, text text, page_start int,
page_end int, spans jsonb, embedding vector(384))`, ordering by
`embedding <=> query_embedding`. Keep the same embedding model, or re-ingest.

## Limitations

- Scanned PDFs without a text layer produce no text (ingest reports a warning); add OCR for those.
- Chunk sizes are in characters, not tokens.
- With free models, citations depend on the model following the `[S#]` instruction.
  Invalid labels are dropped, but a model can still cite a real source that doesn't
  actually support its claim. Claude's native citations are stricter because they
  quote the exact span.
- Tables and multi-column layouts come out in whatever order pypdf extracts them.

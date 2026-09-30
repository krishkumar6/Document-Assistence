# Document Assistant

Upload PDFs, ask questions about them, and get answers that point to the exact
file and page the answer came from.

Live demo: https://document-assistant-26d1.onrender.com (password protected; it
runs on Render's free plan, so the first visit after a while takes about a minute
to wake up).

## What it does

You give it a set of PDFs, such as a policy handbook, a manual or course notes.
It reads them, splits them into chunks and indexes them. When you ask something,
it finds the passages most likely to contain the answer and has an LLM write a
short reply using only those passages. Every claim in the reply links to its
source, like `helios_security_policy.pdf, page 2`, and you can click it to see
the exact text.

There are two ways to ask:

- **Quick answer** searches once and answers. It's fast and good for direct questions.
- **Research** hands the question to an agent with three tools: document search,
  a calculator and web search. It can look things up several times, do the math
  properly instead of guessing, and pull in outside information when the
  documents don't cover it. Every step it takes is shown live and logged, and a
  step limit stops it from looping forever.

Answers stream in word by word, and there's a simple web page for people who
don't want to touch an API.

## What I think is interesting about it

**Citations point to a page, not just a file.** Chunks are allowed to cross page
boundaries (so a sentence split across two pages isn't cut in half), but each
chunk remembers which characters came from which page. When the answer is
written, a chunk is split back into its pages, so a citation always lands on
one exact page. If the model cites a source it was never given, the citation is
dropped instead of shown.

**It doesn't depend on one AI provider.** It runs on Groq's free tier by default,
and also works with Gemini, a local Ollama model or Claude. Embeddings, the
search index and the reranker all run locally, so the whole thing costs nothing
to run.

**The search is hybrid.** Plain vector search is good at meaning but bad at exact
strings like form codes (`NW-IR-3`) or numbers (`3.85`). So it also runs BM25
keyword search, merges the two result lists, and then re-scores the top
candidates with a small cross-encoder model.

**I measured things instead of guessing.** The repo has a labelled question set
(the correct file and page for each question) and scripts that compare chunk
sizes and search strategies. Most of the decisions below came from those numbers.

## What I optimized, with numbers

**Chunk size.** I compared 500-character and 2,000-character chunks. Both gave
correct answers, but small chunks ranked the right page first more often (94% vs
88%) and sent about 2.7x fewer tokens to the model per question. So small is the
default.

**Search quality.** On 26 labelled questions, how often the right page came first:

| Strategy | Small chunks | Large chunks |
|---|---|---|
| Vector search only | 88% | 77% |
| BM25 only | 85% | 88% |
| Hybrid + reranker | 96% | 100% |

The reranker did most of the work. It fixed every reworded question that plain
search missed. BM25 on its own found every exact-code question but struggled
with paraphrases, which is exactly why the two are combined.

**Reranking speed.** Cutting the reranker's candidate pool from 20 to 12 kept the
same accuracy and made it about 35% faster (1.08 s down to 0.61 s per search).

**Running on a tiny free server.** Render's free plan gives 512 MB of RAM and a
tenth of a CPU. Searches took about 12 seconds there at first. The cause was the
model runtime starting one busy-waiting thread per visible core, all fighting
over that small CPU share. Pinning it to a single thread brought searches down
to about 3 seconds. The app uses around 144 MB of memory on that plan.

**Not paying twice for the same question.** Repeat questions are served from an
answer cache: about 0.3 s and zero tokens, instead of around 3 s. Prompts are
also laid out so the provider's own prompt caching can kick in. The fixed
instructions come first, and the agent's conversation only ever grows at the
end, so each step reuses the previous step's prefix.

**Surviving a server that forgets.** The free plan wipes the disk on every
restart. PDFs in the `library/` folder are indexed when the image is built and
restored in seconds on startup, so the core documents are always there.

## Built with

Python, FastAPI, Chroma, MiniLM embeddings, BM25, a MiniLM cross-encoder
reranker (via fastembed), Groq (`gpt-oss-120b`), server-sent events for
streaming, and Docker. Deployed on Render. 69 tests cover chunking, citations,
retrieval, the agent loop, streaming, caching and security.

## Running it locally

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env        # add a free Groq key
mkdir data\pdfs && copy library\*.pdf data\pdfs   # or put your own PDFs there
python -m app.ingest          # index everything in data/pdfs
uvicorn app.main:app --reload # then open http://localhost:8000
```

To reproduce the comparisons: `python -m eval.compare_chunks` and
`python -m eval.compare_retrieval`.

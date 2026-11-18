import json
import re
import sqlite3
import urllib.request
from typing import Any, Callable, Dict, List, Optional
import numpy as np

CORPUS_DB = "weblog_corpus.db"
INDEX_DB = "brain_index.db"
VEC_PATH = "brain_vectors.npy"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
LLM_MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text"

_VECTORS: Optional[np.ndarray] = None


def _load_vectors() -> np.ndarray:
    global _VECTORS
    if _VECTORS is None:
        _VECTORS = np.load(VEC_PATH).astype(np.float32)
    return _VECTORS


def embed_query(text: str) -> np.ndarray:
    payload = {
        "model": EMBED_MODEL,
        "input": [f"search_query: {text}"],
        "keep_alive": "20m",
    }
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        vec = np.array(data["embeddings"][0], dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-12)


def run_corpus_sql(sql_query: str, purpose: str = "") -> Dict[str, Any]:
    cleaned = sql_query.strip().rstrip(";")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return {"error": "Only read-only SELECT or WITH queries are permitted."}
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b"
    if re.search(forbidden, cleaned, re.IGNORECASE):
        return {"error": "Blocked potentially mutating SQL keyword."}

    try:
        conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
        cur = conn.execute(cleaned)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(60)
        conn.close()
        return {
            "purpose": purpose,
            "sql": cleaned,
            "columns": cols,
            "row_count_returned": len(rows),
            "rows": [dict(zip(cols, r)) for r in rows],
        }
    except Exception as e:
        return {"error": f"SQL execution error: {e}", "sql": cleaned}


def search_delong_passages(
    query: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    genres: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    top_k = max(1, min(int(top_k or 5), 15))
    q_vec = embed_query(query)
    mat = _load_vectors()

    conn = sqlite3.connect(f"file:{INDEX_DB}?mode=ro", uri=True)
    where_clauses = []
    params: List[Any] = []

    if year_min is not None:
        where_clauses.append("year >= ? AND year > 0")
        params.append(int(year_min))
    if year_max is not None:
        where_clauses.append("year <= ? AND year > 0")
        params.append(int(year_max))
    if genres:
        placeholders = ",".join("?" for _ in genres)
        where_clauses.append(f"post_genre IN ({placeholders})")
        params.extend(genres)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    valid_rows = conn.execute(
        f"SELECT chunk_idx FROM chunks {where_sql}", params
    ).fetchall()

    if not valid_rows:
        conn.close()
        return {"query": query, "filters": {"year_min": year_min, "year_max": year_max, "genres": genres}, "matches": []}

    cand_indices = np.array([r[0] for r in valid_rows], dtype=np.int64)
    scores = mat[cand_indices] @ q_vec
    ranked_order = np.argsort(-scores)

    matches = []
    per_post_counts: Dict[int, int] = {}
    for pos in ranked_order:
        c_idx = int(cand_indices[pos])
        score = float(scores[pos])
        row = conn.execute(
            """
            SELECT post_id, year, post_date, corpus_era, post_genre, title, wayback_url, chunk_text
            FROM chunks WHERE chunk_idx = ?
            """,
            (c_idx,),
        ).fetchone()
        if not row:
            continue
        pid, year, pdate, era, genre, title, wb_url, ctext = row
        if per_post_counts.get(pid, 0) >= 2:
            continue
        per_post_counts[pid] = per_post_counts.get(pid, 0) + 1
        matches.append({
            "rank": len(matches) + 1,
            "similarity": round(score, 4),
            "post_id": pid,
            "date": pdate or era,
            "genre": genre,
            "title": title,
            "wayback_url": wb_url if wb_url else f"Archive ID #{pid}",
            "excerpt": ctext[:900] + ("..." if len(ctext) > 900 else ""),
            "full_excerpt": ctext,
        })
        if len(matches) >= top_k:
            break

    conn.close()
    return {
        "query": query,
        "filters": {"year_min": year_min, "year_max": year_max, "genres": genres},
        "matches": matches,
    }


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_corpus_sql",
            "description": (
                "Execute a read-only SQLite SELECT query on weblog_corpus.db to compute exact counts, "
                "yearly frequency trends, or metadata lookups. Table `posts` columns: "
                "id, title, exact_post_datetime (ISO string), corpus_era ('pre-2005' or 'post-2005'), "
                "post_genre ('essay_or_column', 'quote_commentary', 'clipping_or_mail', 'teaching_or_meta', "
                "'link_roundup', 'archive_dump', 'corrupt_binary'), is_unique_corpus (1=deduplicated valid post, 0=excluded), "
                "wayback_url, body_clean. Always filter `is_unique_corpus = 1` and search text using `body_clean` (NOT query_clean). SQLITE RULES: (1) For yearly trends, ALWAYS write ONE single `GROUP BY year` query—NEVER emit multiple single-year queries. (2) `strftime('%Y', exact_post_datetime)` returns TEXT, so either compare with strings (`>= '2008'`) or wrap in `CAST(strftime('%Y', exact_post_datetime) AS INTEGER)`. (3) If this tool returns an `error`, immediately invoke `run_corpus_sql` again with fixed SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "Valid SQLite SELECT query."},
                    "purpose": {"type": "string", "description": "Brief description of what this query measures."},
                },
                "required": ["sql_query", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_delong_passages",
            "description": (
                "Semantic vector search across J. Bradford DeLong's deduplicated core writings "
                "(14,103 posts across essay_or_column, quote_commentary, teaching_or_meta). "
                "For longitudinal comparisons across decades, call this tool multiple times with distinct "
                "year_min / year_max ranges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Detailed English semantic search query."},
                    "year_min": {"type": "integer", "description": "Optional start year (inclusive), e.g. 1998."},
                    "year_max": {"type": "integer", "description": "Optional end year (inclusive), e.g. 2007."},
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional genre filter subset of ['essay_or_column', 'quote_commentary', 'teaching_or_meta'].",
                    },
                    "top_k": {"type": "integer", "description": "Number of passages to retrieve (default 8, max 15)."},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are **DeLong Brain**, a strict, zero-hallucination archival research assistant analyzing economist J. Bradford DeLong's corpus.

### NON-NEGOTIABLE EPISTEMIC & CITATION RULES
1. **Evidence-Only Synthesis**: Synthesize ALL relevant passages returned by `search_delong_passages` alongside any `run_corpus_sql` statistics. NEVER ignore retrieved passages, and NEVER invent dates, titles, URLs, or facts.
2. **Strict Inline Citation Format (CRITICAL)**:
   - NEVER write vague phrases like "in a 2005 post" or "in a 2018 article" without the exact citation bracket.
   - EVERY single claim or quote MUST be followed immediately by the exact bracket copied from tool outputs:
     `[date | title | wayback_url]`
   - Good example: `DeLong praised Greenspan's tactical flexibility [2005-03-14 | Alan Greenspan's Legacy | https://web.archive.org/...].`
   - Bad example (FORBIDDEN): `In a 2005 post, DeLong praised Greenspan.`
3. **Explicit Refusal on Missing Evidence**: If retrieved passages do not directly address a topic or time period, state clearly: *"The indexed corpus does not contain direct evidence on [topic/period]."*
4. **Era Search Strategy**: Note that indexed dates are dense in 1995–2005 and 2017–2024, with sparse dated coverage in 2006–2016. When researching post-2008 topics, search `year_min=2008` WITHOUT capping `year_max` at 2012 so retrospective 2017+ posts are retrieved.
5. **Corpus Coverage Note (MANDATORY FINAL SECTION)**:
   - End every response with `### Corpus Coverage Note`.
   - Explicitly report any multi-year archival gaps revealed by SQL counts (e.g., if 2006–2016 has zero or sparse indexed records, state clearly that post-crisis insights rely on retrospective posts from 2017+ rather than contemporaneous 2008–2012 posts).
   - State the active scope: deduplicated `is_unique_corpus = 1` (14,103 core semantic posts; 2,470 bulk archive dumps and 272 corrupted files excluded).
6. **Concise Analytical Synthesis (CRITICAL)**:
   - NEVER paste raw multi-sentence excerpts verbatim. Summarize DeLong's argument or commentary in 1–2 concise sentences per point, quoting at most a short key phrase.
   - When comparing two eras (e.g., Pre-2005 vs. Post-2017), you MUST include a dedicated subsection for BOTH eras. If one era's `search_delong_passages` returns 0 matches, explicitly note that those SQL-counted posts fall outside the indexed core genres (`essay_or_column`, `quote_commentary`, `teaching_or_meta`).
   - Do NOT attach `[date | title | url]` brackets to SQL `COUNT(*)` statistics—only attach them to specific post claims."""


def ask_delong_brain(
    question: str,
    max_rounds: int = 5,
    on_step: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_trace: List[Dict[str, Any]] = []
    collected_sources: Dict[int, Dict[str, Any]] = {}
    did_cross_era_rewrite = False
    force_no_tools = False

    for round_idx in range(1, max_rounds + 1):
        if on_step:
            on_step(f"Round {round_idx}: Reasoning with {LLM_MODEL}...")
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "tools": [] if force_no_tools else TOOLS_SCHEMA,
            "stream": False,
            "keep_alive": "20m",
            "options": {"temperature": 0.1, "num_ctx": 16384, "num_predict": 2048},
        }
        req = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            res = json.loads(resp.read().decode("utf-8"))

        msg = res.get("message", {})
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # 若本轮检索到了多年代史料，但模型回复中完全未引用早期年份，强制做一次全证据对齐校验
            early_srcs = [s for s in collected_sources.values() if str(s.get("date", ""))[:4].isdigit() and int(str(s["date"])[:4]) <= 2006]
            late_srcs = [s for s in collected_sources.values() if str(s.get("date", ""))[:4].isdigit() and int(str(s["date"])[:4]) >= 2007]
            content = msg.get("content", "").strip()
            if (
                not did_cross_era_rewrite
                and early_srcs
                and late_srcs
                and not any(str(s["date"])[:4] in content for s in early_srcs)
            ):
                did_cross_era_rewrite = True
                force_no_tools = True
                if on_step:
                    on_step("Synthesizing omitted pre-2007 evidence...")
                digest = json.dumps(list(collected_sources.values()), ensure_ascii=False)
                messages.append({
                    "role": "user",
                    "content": (
                        "You omitted the retrieved pre-2007 passages in your answer. "
                        "Do NOT call any more tools. Rewrite the complete response right now so BOTH the pre-2007 era and post-2008 era "
                        "are synthesized with exact inline citations from this full retrieved evidence list:\n"
                        f"{digest}"
                    ),
                })
                continue
            return {
                "question": question,
                "answer": content,
                "tool_trace": tool_trace,
                "sources": list(collected_sources.values()),
            }

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)

            if on_step:
                on_step(f"Executing `{name}` with {args}")

            if name == "search_delong_passages" and "sql_query" in args:
                name = "run_corpus_sql"
            elif name == "run_corpus_sql" and "query" in args and "sql_query" not in args:
                name = "search_delong_passages"

            try:
                if name == "run_corpus_sql":
                    result = run_corpus_sql(**args)
                elif name == "search_delong_passages":
                    result = search_delong_passages(**args)
                    for m in result.get("matches", []):
                        collected_sources[m["post_id"]] = m
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except Exception as tool_err:
                result = {"error": f"Invalid arguments for {name}: {tool_err}"}

            tool_trace.append({"tool": name, "args": args, "result_summary": result})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    if collected_sources:
        if on_step:
            on_step("Finalizing synthesis from collected sources...")
        digest = json.dumps(list(collected_sources.values()), ensure_ascii=False)
        messages.append({
            "role": "user",
            "content": f"Stop calling tools and write the final evidence-backed answer with inline citations and Corpus Coverage Note based on:\n{digest}",
        })
        final_payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "20m",
            "options": {"temperature": 0.1, "num_ctx": 16384, "num_predict": 2048},
        }
        req = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=json.dumps(final_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        return {
            "question": question,
            "answer": res.get("message", {}).get("content", "").strip(),
            "tool_trace": tool_trace,
            "sources": list(collected_sources.values()),
        }

    return {
        "question": question,
        "answer": "Reached maximum tool reasoning rounds before finalizing answer.",
        "tool_trace": tool_trace,
        "sources": list(collected_sources.values()),
    }


if __name__ == "__main__":
    test_q = (
        "How did Brad DeLong's view on Alan Greenspan and Federal Reserve monetary policy "
        "change between the pre-2007 era and the post-2008 financial crisis era? "
        "Also include yearly mention counts from SQL."
    )
    print(f"Running test query:\n{test_q}\n" + "=" * 60, flush=True)
    out = ask_delong_brain(test_q, on_step=lambda s: print(f"  [Step] {s}", flush=True))
    print("\n" + "=" * 60 + "\n[Final Synthesis]:\n")
    print(out["answer"])

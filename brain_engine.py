import json
import os
import re
import sqlite3
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def _resolve_db_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    alt = os.path.join("database", filename)
    if os.path.exists(alt):
        return alt
    return filename


CORPUS_DB = _resolve_db_path("weblog_corpus.db")
INDEX_DB = _resolve_db_path("brain_index.db")
VEC_PATH = _resolve_db_path("brain_vectors.npy")

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
LLM_MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text"

CORE_VOICE_GENRES = ["essay_or_column", "quote_commentary"]

_VECTORS: Optional[np.ndarray] = None
_YEARS_BACKFILLED = False

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "about", "into", "through", "during", "before", "after",
    "between", "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "brad", "delong", "delongs", "think", "view", "views", "write", "wrote",
    "post", "posts", "era", "versus", "vs", "include", "yearly", "counts", "sql",
    "perspective", "perspectives", "opinion", "opinions",
}

_MONTHS_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_NAV_TAIL_RE = re.compile(r"«\s*[^»\n]{1,220}»\s*", re.DOTALL)
_FOOTER_SIG_RE = re.compile(
    r"Posted by DeLong at\s+[0-9:]+\s*[AP]M\s*\|\s*Permanent Link"
    r"(?:\s*\|\s*Comments\s*\(\d+\))?(?:\s*\|\s*Trackback\s*\(\d+\))?",
    re.IGNORECASE,
)


def _clean_excerpt_for_llm(raw_chunk: str, title: str = "") -> str:
    """Strip stale chunk headers, Movable Type nav banners, and blog signature footers."""
    text = raw_chunk.strip()
    text = re.sub(r"^\[[^\]]+\]\s*[^\n]*\n+", "", text, count=1)
    text = re.sub(r"^.*?Website Policies\s*", "", text, count=1, flags=re.IGNORECASE | re.DOTALL)
    if "«" in text[:450] and "»" in text[:550]:
        text = _NAV_TAIL_RE.sub("", text, count=1)
        text = re.sub(
            rf"^(?:.*?(?:Semi-Daily Journal|Reality-Based Community|DeLong's Home Page|Main Weblog Page|Main Website Page|Rules of the Road)\s*)+",
            "",
            text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(rf"^{_MONTHS_PAT}\s+\d{{1,2}},\s+\d{{4}}\s*", "", text, count=1)
    if title:
        clean_t = re.sub(
            r":?\s*Archive Entry From Brad DeLong[’']s (?:Webjournal|Semi-Daily Journal)\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
        if clean_t and text.lower().startswith(clean_t.lower()):
            text = text[len(clean_t) :].lstrip(" :\n\t")
    text = _FOOTER_SIG_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _smart_display_title(raw_title: str, cleaned_text: str) -> str:
    """Strip legacy blog prefixes and replace generic 'Untitled TypePad Entry' with a real headline."""
    t = (raw_title or "").strip()
    t = re.sub(
        r":?\s*Archive Entry From Brad DeLong[’']s (?:Webjournal|Semi-Daily Journal)\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    t = re.sub(
        r"^Brad DeLong[’']s (?:Website|Semi-Daily Journal(?:\s*:\s*A Weblog)?)\s*:\s*",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()

    if t and t.lower() not in {"untitled", "untitled typepad entry", "untitled entry"}:
        return t

    body = re.sub(
        r"^(?:Things that went whizzing by that I want to remember\.*|Very Briefly Noted:|Paragraphs)\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    ).strip()
    first_line = next((ln.strip(" :-•1234567890.)") for ln in body.splitlines() if len(ln.strip()) >= 12), "")
    if first_line:
        snippet = re.split(r"(?<=[.!?])\s+|:\s+", first_line, maxsplit=1)[0].strip()
        if len(snippet) > 62:
            snippet = snippet[:59].rsplit(" ", 1)[0] + "..."
        if len(snippet) >= 10:
            return snippet
    return "Semi-Daily Journal Note"


def _title_to_flexible_regex(title: str) -> str:
    """Escape title while allowing straight/curly apostrophes."""
    parts = re.split(r"['’]", title.strip())
    return r"['’]".join(re.escape(p) for p in parts)


def _polish_chatbot_output(text: str, sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """Convert raw URLs / [Title] / plain cited titles into clickable Markdown links or clean italics."""
    # 1. Strip raw `Archive ID #123` artifacts
    text = re.sub(r"\s*\(Archive ID #\d+\)", "", text)
    text = re.sub(r",?\s*Archive ID #\d+", "", text)
    text = re.sub(r"\[([^\]]+)\]\(Archive ID #\d+\)", r"*\1*", text)

    # 2. Upgrade ([Title], date) / (Title, date) -> ([Title](wayback_url), date) or (*Title*, date)
    if sources:
        for s in sorted(sources, key=lambda x: len(str(x.get("title", ""))), reverse=True):
            title = str(s.get("title", "")).strip()
            wb_url = str(s.get("wayback_url", "")).strip()
            if not title:
                continue
            flex_title = _title_to_flexible_regex(title)
            if wb_url.startswith("http"):
                esc_url = re.escape(wb_url)
                # Case A: ("Title," https://..., date)
                text = re.sub(
                    rf'\(\s*[\["“\*]*{flex_title}[\],"”\*]*\s*{esc_url}\s*,',
                    f"([{title}]({wb_url}),",
                    text,
                    flags=re.IGNORECASE,
                )
                # Case B: ([Title], date) or (Title, date) or (*Title*, date)
                text = re.sub(
                    rf'\(\s*[\["“\*]*{flex_title}[\],"”\*]*\s*,',
                    f"([{title}]({wb_url}),",
                    text,
                    flags=re.IGNORECASE,
                )
            else:
                # No Wayback URL: turn ([Title], date) or (Title, date) into (*Title*, date)
                text = re.sub(
                    rf'\(\s*[\["“\*]*{flex_title}[\],"”\*]*\s*,',
                    f"(*{title}*,",
                    text,
                    flags=re.IGNORECASE,
                )

    # 3. Clean any remaining unlinked ([Any Title], date) into (*Any Title*, date) so brackets never leak
    text = re.sub(r"\(\[([^\]]+)\](?!\()\s*,", r"(*\1*,", text)

    # Fallback: catch any remaining raw `("Title," https://..., year)`
    text = re.sub(
        r'\(\s*["“\*]*([^"”\(\)]+?)[,"”\*]+\s*(https?://[^\s,\)]+)\s*,',
        r"([\1](\2),",
        text,
    )

    # 4. Strip rigid standalone era headers
    text = re.sub(
        r"(?m)^(?:#{1,4}\s*|\*\*)?(?:Pre-2005 Era|Post-2005 Era|Core Thesis|Corpus Coverage Note)(?:\*\*)?:?\s*\n+",
        "",
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_vectors() -> np.ndarray:
    global _VECTORS
    if _VECTORS is None:
        _VECTORS = np.load(VEC_PATH).astype(np.float32)
    return _VECTORS


def _ensure_index_years_backfilled() -> None:
    """One-time lightweight backfill of missing years & full dates in brain_index.db."""
    global _YEARS_BACKFILLED
    if _YEARS_BACKFILLED or not os.path.exists(INDEX_DB) or not os.path.exists(CORPUS_DB):
        return
    _YEARS_BACKFILLED = True
    try:
        idx = sqlite3.connect(INDEX_DB)
        needs_fix = idx.execute(
            "SELECT COUNT(*) FROM chunks WHERE year = 0 OR (post_date LIKE '%,')"
        ).fetchone()[0]
        if needs_fix == 0:
            idx.close()
            return

        main_conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
        updates: List[Tuple[int, str, int]] = []
        for pid, dt, pdate, rpath in main_conn.execute(
            "SELECT id, exact_post_datetime, post_date, rel_path FROM posts WHERE is_unique_corpus = 1"
        ):
            raw_date = (dt or "")[:10] or (pdate or "").strip()
            if raw_date.lower() == "unknown":
                raw_date = ""
            m = re.search(r"(199\d|20[0-2]\d)", raw_date) or re.search(r"(199\d|20[0-2]\d)", rpath or "")
            yr = int(m.group(1)) if m else 0
            if yr > 0:
                clean_d = raw_date or str(yr)
                updates.append((yr, clean_d, pid))
        main_conn.close()

        if updates:
            idx.executemany(
                """
                UPDATE chunks
                SET year = CASE WHEN year = 0 THEN ? ELSE year END,
                    post_date = CASE
                        WHEN post_date = '' OR post_date IS NULL OR post_date LIKE '%,' THEN ?
                        ELSE post_date
                    END
                WHERE post_id = ? AND (year = 0 OR post_date LIKE '%,')
                """,
                updates,
            )
            idx.commit()
        idx.close()
    except Exception:
        pass


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


def _build_fts_query(query: str) -> str:
    """Convert natural language query into a clean SQLite FTS5 OR expression."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", query)
    keywords = [t for t in tokens if len(t) >= 2 and t.lower() not in _STOPWORDS]
    if not keywords:
        return ""
    seen = set()
    uniq = []
    for k in keywords:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            uniq.append(f'"{k}"')
        if len(uniq) >= 8:
            break
    return " OR ".join(uniq)


def _fts_lexical_ranks(query: str, limit: int = 150) -> Dict[int, int]:
    """Return {post_id: 1-based BM25 rank} using posts_fts for exact proper-noun/acronym recall."""
    fts_q = _build_fts_query(query)
    if not fts_q or not os.path.exists(CORPUS_DB):
        return {}
    try:
        conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
        rows = conn.execute(
            """
            SELECT p.id
            FROM posts_fts
            JOIN posts p ON posts_fts.rowid = p.id
            WHERE posts_fts MATCH ?
              AND p.is_unique_corpus = 1
            ORDER BY bm25(posts_fts)
            LIMIT ?
            """,
            (fts_q, limit),
        ).fetchall()
        conn.close()
        return {int(r[0]): idx + 1 for idx, r in enumerate(rows)}
    except Exception:
        return {}


def _fetch_quote_ratios(post_ids: List[int]) -> Dict[int, float]:
    if not post_ids or not os.path.exists(CORPUS_DB):
        return {}
    try:
        conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
        placeholders = ",".join("?" for _ in post_ids)
        rows = conn.execute(
            f"SELECT id, COALESCE(quote_ratio, 0.0) FROM posts WHERE id IN ({placeholders})",
            post_ids,
        ).fetchall()
        conn.close()
        return {int(r[0]): float(r[1]) for r in rows}
    except Exception:
        return {}


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


def _search_single_slice(
    conn: sqlite3.Connection,
    mat: np.ndarray,
    q_vec: np.ndarray,
    fts_post_ranks: Dict[int, int],
    y_min: Optional[int],
    y_max: Optional[int],
    active_genres: List[str],
    limit_k: int,
) -> List[Dict[str, Any]]:
    where_clauses: List[str] = []
    params: List[Any] = []

    allow_undated_pre2005 = (y_min is None or y_min <= 2004)

    if y_min is not None and y_max is not None:
        if allow_undated_pre2005:
            where_clauses.append("((year BETWEEN ? AND ?) OR (year = 0 AND corpus_era = 'pre-2005'))")
        else:
            where_clauses.append("(year BETWEEN ? AND ?)")
        params.extend([y_min, y_max])
    elif y_min is not None:
        if allow_undated_pre2005:
            where_clauses.append("(year >= ? OR (year = 0 AND corpus_era = 'pre-2005'))")
        else:
            where_clauses.append("year >= ?")
        params.append(y_min)
    elif y_max is not None:
        if allow_undated_pre2005:
            where_clauses.append("((year > 0 AND year <= ?) OR (year = 0 AND corpus_era = 'pre-2005'))")
        else:
            where_clauses.append("(year > 0 AND year <= ?)")
        params.append(y_max)

    if active_genres:
        placeholders = ",".join("?" for _ in active_genres)
        where_clauses.append(f"post_genre IN ({placeholders})")
        params.extend(active_genres)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    valid_rows = conn.execute(
        f"SELECT chunk_idx, post_id, year FROM chunks {where_sql}", params
    ).fetchall()

    if not valid_rows:
        return []

    cand_indices = np.array([r[0] for r in valid_rows], dtype=np.int64)
    cand_post_ids = [int(r[1]) for r in valid_rows]
    cand_years = [int(r[2]) for r in valid_rows]

    vec_scores = mat[cand_indices] @ q_vec
    vec_order = np.argsort(-vec_scores)

    vec_rank_map: Dict[int, int] = {}
    for rank_idx, pos in enumerate(vec_order[:200], start=1):
        vec_rank_map[int(pos)] = rank_idx

    pool_positions = set(vec_rank_map.keys())
    if fts_post_ranks:
        for pos, pid in enumerate(cand_post_ids):
            if pid in fts_post_ranks:
                pool_positions.add(pos)

    pool_post_ids = list({cand_post_ids[pos] for pos in pool_positions})
    quote_ratios = _fetch_quote_ratios(pool_post_ids)

    scored_pool: List[Tuple[float, float, int]] = []
    for pos in pool_positions:
        pid = cand_post_ids[pos]
        v_sim = float(vec_scores[pos])
        r_vec = vec_rank_map.get(pos)
        r_lex = fts_post_ranks.get(pid)

        rrf = (1.0 / (60.0 + r_vec) if r_vec is not None else 0.0) + (
            0.85 / (60.0 + r_lex) if r_lex is not None else 0.0
        )
        qr = quote_ratios.get(pid, 0.0)
        orig_boost = 1.0 + 0.12 * max(0.0, 1.0 - min(qr, 1.0))
        scored_pool.append((rrf * orig_boost, v_sim, pos))

    scored_pool.sort(key=lambda x: -x[0])

    selected_positions: List[Tuple[float, int]] = []
    seen_posts = set()
    per_year_counts: Dict[int, int] = {}

    for max_per_year in (1, 2):
        if len(selected_positions) >= limit_k:
            break
        for _, v_sim, pos in scored_pool:
            if len(selected_positions) >= limit_k:
                break
            pid = cand_post_ids[pos]
            yr = cand_years[pos]
            if pid in seen_posts:
                continue
            if per_year_counts.get(yr, 0) >= max_per_year:
                continue
            seen_posts.add(pid)
            per_year_counts[yr] = per_year_counts.get(yr, 0) + 1
            selected_positions.append((v_sim, pos))

    selected_positions.sort(key=lambda item: (cand_years[item[1]] or 2004, -item[0]))

    matches = []
    for v_sim, pos in selected_positions:
        c_idx = int(cand_indices[pos])
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
        qr = round(quote_ratios.get(pid, 0.0), 2)
        cleaned_text = _clean_excerpt_for_llm(ctext, title)
        if pdate and year and str(year) not in pdate:
            date_label = f"{pdate} {year}"
        else:
            date_label = pdate or (str(year) if year else era)
        display_title = _smart_display_title(title, cleaned_text)
        attribution_hint = (
            f"Heavy external quotation (quote_ratio={qr}) — separate quoted text from DeLong's own commentary"
            if qr >= 0.45
            else f"Mostly DeLong's own words (quote_ratio={qr})"
        )
        matches.append({
            "similarity": round(v_sim, 4),
            "post_id": pid,
            "year": year,
            "date": date_label,
            "genre": genre,
            "quote_ratio": qr,
            "attribution_note": attribution_hint,
            "title": display_title,
            "wayback_url": wb_url if wb_url else f"Archive ID #{pid}",
            "excerpt": cleaned_text[:700] + ("..." if len(cleaned_text) > 700 else ""),
            "full_excerpt": cleaned_text,
        })
    return matches


def search_delong_passages(
    query: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    genres: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Unified all-years hybrid search with per-year diversity spreading."""
    _ensure_index_years_backfilled()
    top_k = max(1, min(int(top_k or 10), 15))
    active_genres = genres if genres else CORE_VOICE_GENRES

    q_vec = embed_query(query)
    mat = _load_vectors()
    fts_post_ranks = _fts_lexical_ranks(query, limit=150)

    conn = sqlite3.connect(f"file:{INDEX_DB}?mode=ro", uri=True)
    y_min = int(year_min) if year_min is not None else None
    y_max = int(year_max) if year_max is not None else None

    merged = _search_single_slice(conn, mat, q_vec, fts_post_ranks, y_min, y_max, active_genres, top_k)
    conn.close()

    for idx, m in enumerate(merged, start=1):
        m["rank"] = idx

    return {
        "query": query,
        "filters": {"year_min": y_min, "year_max": y_max, "genres": active_genres},
        "matches": merged,
    }


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_corpus_sql",
            "description": (
                "Execute a read-only SQLite SELECT query on weblog_corpus.db to compute exact counts, "
                "yearly frequency trends, or metadata lookups.\n"
                "EXACT `posts` COLUMNS (NO OTHER COLUMNS EXIST — NEVER write `year_min` or `year_max` in SQL!): "
                "id, title, post_date, exact_post_datetime (ISO string, NULL for ~3.5k pre-2005 posts), "
                "corpus_era ('pre-2005' or 'typepad'), post_genre ('essay_or_column', 'quote_commentary', "
                "'teaching_or_meta', 'clipping_or_mail', 'link_roundup'), quote_ratio (FLOAT 0.0-1.0), "
                "own_words (INT), is_unique_corpus (1=valid deduplicated doc), is_canonical_post (1/0), "
                "wayback_url, rel_path, body_clean.\n"
                "FULL-TEXT INDEX `posts_fts`: Virtual FTS5 table linked via `JOIN posts_fts ON posts.id = posts_fts.rowid`. "
                "CRITICAL FTS5 SYNTAX: Spaces mean implicit AND! Always separate alternative keywords with uppercase `OR` "
                "(e.g. `posts_fts MATCH '\"growth model\" OR Solow OR \"endogenous growth\"'`), NEVER a long space-separated string.\n"
                "SQLITE RULES: (1) Always filter `is_unique_corpus = 1`. (2) `strftime('%Y', exact_post_datetime)` returns TEXT; "
                "cast with `CAST(strftime('%Y', exact_post_datetime) AS INTEGER)` or compare strings (`>= '2008'`). "
                "(3) Never repeat an identical SQL query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "Valid SQLite SELECT query using only actual columns of `posts` and `posts_fts`."},
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
                "Unified hybrid semantic + BM25 search across ALL years (1995–present) of J. Bradford DeLong's "
                "deduplicated writings, automatically spreading top results across every relevant year in chronological order. "
                "Omit `year_min` and `year_max` unless the user explicitly asks about a specific date range. "
                "Defaults to DeLong Core Voice (`essay_or_column` and `quote_commentary`, 12,599 posts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Detailed English search query combining key concepts and proper nouns."},
                    "year_min": {"type": "integer", "description": "Optional start year (inclusive) ONLY if user specifies a date filter."},
                    "year_max": {"type": "integer", "description": "Optional end year (inclusive) ONLY if user specifies a date filter."},
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional genre filter. Omit to use default ['essay_or_column', 'quote_commentary'].",
                    },
                    "top_k": {"type": "integer", "description": "Number of passages to retrieve across years (default 10, max 15)."},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are **DeLong Brain**, a sharp, conversational AI research assistant with direct access to economist J. Bradford DeLong's weblog archive (1995–present).

### NON-NEGOTIABLE GROUNDING & CITATION RULES
1. **ALWAYS SEARCH FIRST (ZERO-MEMORY RULE)**: You have NO built-in knowledge of DeLong's posts. On EVERY user question, you MUST call `search_delong_passages` BEFORE writing any answer. NEVER answer without tool results, and NEVER invent titles, dates, or URLs.
2. **MANDATORY PARENTHETICAL INLINE CITATIONS**: Every single claim or reference to a post MUST include an inline parenthetical citation copied from the tool outputs:
   - When `wayback_url` starts with `http`: `([Title](wayback_url), date)`
   - When `wayback_url` starts with `Archive ID` (NOT a URL): `(*Title*, date)` — NEVER write `Archive ID` or raw `http` URLs outside `[Title](url)`!
   - NEVER omit parenthetical citations! Even when mentioning a date in prose, always attach `([Title](wayback_url), date)` or `(*Title*, date)` at the end of the clause.
3. **Separate DeLong's voice from blockquotes (`quote_ratio`)**: In `quote_commentary` posts (especially `quote_ratio >= 0.45`), large sections are quoted from other writers. Never attribute an external quote to DeLong himself—say *"DeLong quotes [Author] arguing that..."* and then share DeLong's own take.

### HOW TO RESPOND (NATURAL CHATBOT PROSE)
- **Group related years into 3–4 cohesive thematic paragraphs** (e.g., grouping 1996–2004 together, 2017–2018 together, and 2019–2021 together)—do NOT write one single-sentence paragraph per year, do NOT output a numbered list (`1. 2. 3.`), and do NOT open with robotic filler like *"Here are some key points from his writings:"*.
- **Follow the chronological arc naturally** across whatever years appear in the search results without using artificial section headers like `Pre-2005 Era` or `Post-2005 Era`.
- **Mention archival coverage only when relevant**: Dated posts are dense in 1995–2005 and 2017–2024, and sparse in 2006–2016. Only mention this briefly in a natural closing sentence if the user asks about 2006–2016 or full-timeline counts.
- **Be honest about gaps**: If neither semantic search nor SQL finds relevant evidence, say so directly in plain language without guessing. Never report `top_k` chunk counts as total corpus post counts.

### EXAMPLE OF IDEAL CHATBOT RESPONSE
User: *"What are Brad DeLong's core arguments about macroeconomic growth models?"*
Assistant:
Brad DeLong generally treats the standard Solow growth model as an essential pedagogical baseline, while consistently arguing that it falls short when explaining real-world growth miracles and long-run productivity shifts.

In his early writings, DeLong emphasized that capital accumulation alone cannot account for rapid post-WWII catch-up growth in economies like Japan, pointing instead to technology transfer, institutional reform, and endogenous innovation ([Endogenous Growth](https://web.archive.org/...), April 29, 1999). Examining the U.S. productivity boom of the late 1990s, he similarly noted that traditional convergence dynamics fail to explain sustained accelerations driven by general-purpose technologies ([THE SOLOW PRODUCTIVITY PARADOX IN HISTORICAL PERSPECTIVE](https://web.archive.org/...), pre-2005).

In later posts, DeLong also highlighted the empirical limits of first-generation endogenous growth theory—for instance, quoting Charles I. Jones's critique that R&D-based models struggle to square permanent policy shifts with relatively stable long-run U.S. growth rates (*BRIEFLY NOTED: 2021-03-22 Mo*, 2021-03-22). At the same time, he maintained that simplified textbook models remain useful for training students how to think systematically, provided economists don't mistake those abstractions for the full story (*After the Examination All Professors Are Sad*, 2018-05-17).
"""


def _compact_sources_for_llm(sources: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in s.items() if k != "full_excerpt"}
        for s in sources[:limit]
    ]


def _count_parenthetical_citations(content: str, sources: List[Dict[str, Any]]) -> int:
    """Count how many distinct retrieved sources are properly cited inside parentheses."""
    cited = 0
    for s in sources:
        title_prefix = str(s.get("title", ""))[:22]
        wb_url = str(s.get("wayback_url", ""))
        if not title_prefix:
            continue
        has_paren_title = (
            f"({title_prefix}" in content
            or f"(*{title_prefix}" in content
            or f"([{title_prefix}" in content
            or f'("{title_prefix}' in content
            or (wb_url.startswith("http") and wb_url in content)
        )
        if has_paren_title:
            cited += 1
    return cited


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
    seen_sql_queries = set()
    did_polish_rewrite = False
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
            "options": {"temperature": 0.15, "num_ctx": 16384, "num_predict": 2048},
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
            if not tool_trace and round_idx < max_rounds:
                if on_step:
                    on_step("Enforcing archive search before answering...")
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP. You did not call any search tool yet, so any title or date you write would be hallucinated. "
                        "Call `search_delong_passages` right now to retrieve real passages across all years."
                    ),
                })
                continue

            content = msg.get("content", "").strip()
            src_list = list(collected_sources.values())
            min_required = min(2, len(src_list))
            cited_count = _count_parenthetical_citations(content, src_list)

            years_present = [int(s["year"]) for s in src_list if s.get("year")]
            has_early = any(y <= 2005 for y in years_present)
            has_late = any(y >= 2015 for y in years_present)
            early_cited = _count_parenthetical_citations(
                content, [s for s in src_list if s.get("year") and int(s["year"]) <= 2005]
            )
            missing_early = has_early and has_late and (early_cited == 0)

            if not did_polish_rewrite and src_list and (cited_count < min_required or missing_early):
                did_polish_rewrite = True
                force_no_tools = True
                if on_step:
                    on_step("Adding inline parenthetical citations across retrieved years...")
                compact = _compact_sources_for_llm(src_list, limit=10)
                digest = json.dumps(compact, ensure_ascii=False)
                messages.append({
                    "role": "user",
                    "content": (
                        "Rewrite your response into 3–4 cohesive, conversational chatbot paragraphs (group nearby years "
                        "together into multi-sentence paragraphs; NO single-sentence paragraphs, NO numbered lists, "
                        "NO standalone era headers, and do NOT call any tools). "
                        "Every specific point MUST end with an inline parenthetical citation copied from the sources below: "
                        "use `([Title](wayback_url), date)` when `wayback_url` starts with `http`, or `(*Title*, date)` "
                        "when it starts with `Archive ID`:\n"
                        f"{digest}"
                    ),
                })
                continue
            return {
                "question": question,
                "answer": _polish_chatbot_output(content, src_list),
                "tool_trace": tool_trace,
                "sources": src_list,
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
                    norm_sql = re.sub(r"\s+", " ", args.get("sql_query", "").strip().rstrip(";")).lower()
                    if norm_sql in seen_sql_queries:
                        result = {
                            "warning": "Duplicate SQL query skipped; use the result already returned above and proceed to synthesis."
                        }
                    else:
                        seen_sql_queries.add(norm_sql)
                        result = run_corpus_sql(**args)
                    llm_result = result
                elif name == "search_delong_passages":
                    result = search_delong_passages(**args)
                    for m in result.get("matches", []):
                        collected_sources[m["post_id"]] = m
                    llm_result = {
                        "query": result["query"],
                        "filters": result["filters"],
                        "matches": _compact_sources_for_llm(result.get("matches", []), limit=15),
                    }
                else:
                    result = {"error": f"Unknown tool: {name}"}
                    llm_result = result
            except Exception as tool_err:
                result = {"error": f"Invalid arguments for {name}: {tool_err}"}
                llm_result = result

            tool_trace.append({"tool": name, "args": args, "result_summary": result})
            messages.append({"role": "tool", "content": json.dumps(llm_result, ensure_ascii=False)})

    if collected_sources:
        if on_step:
            on_step("Finalizing response from collected sources...")
        src_list = list(collected_sources.values())
        digest = json.dumps(_compact_sources_for_llm(src_list, limit=10), ensure_ascii=False)
        messages.append({
            "role": "user",
            "content": (
                "Stop calling tools and write 3–4 cohesive, conversational paragraphs (group nearby years together; "
                "NO single-sentence paragraphs, NO numbered lists, NO standalone era headers) "
                "weaving in inline parenthetical citations `([Title](wayback_url), date)` when `wayback_url` starts with `http`, "
                "or `(*Title*, date)` when it starts with `Archive ID`, based on:\n"
                f"{digest}"
            ),
        })
        final_payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "20m",
            "options": {"temperature": 0.15, "num_ctx": 16384, "num_predict": 2048},
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
            "answer": _polish_chatbot_output(res.get("message", {}).get("content", ""), src_list),
            "tool_trace": tool_trace,
            "sources": src_list,
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
        "change over time? Also include yearly mention counts from SQL."
    )
    print(f"Running test query:\n{test_q}\n" + "=" * 60, flush=True)
    out = ask_delong_brain(test_q, on_step=lambda s: print(f"  [Step] {s}", flush=True))
    print("\n" + "=" * 60 + "\n[Final Synthesis]:\n")
    print(out["answer"])
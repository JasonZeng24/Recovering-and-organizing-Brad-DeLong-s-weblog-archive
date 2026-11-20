# Brad DeLong's Weblog Archive

Recovering and organizing Brad DeLong's weblog archive — a Streamlit app built in UC Berkeley's CDSS Data Science Discovery practicum, covering the pre-2005 Movable Type archive and the TypePad era (2005–present).

**Live demo:** https://dsdiscovery-pre2005demo.streamlit.app/

## The problem

DeLong's Semi-Daily Journal spans decades, but surviving pre-2005 copies are scattered across overlapping backup folders, aggregator index pages, and early static course directories, while many referenced figures were never bundled locally. This project consolidates the archive into a deduplicated, searchable database with verified Wayback Machine provenance, genre classification, and missing-asset recovery.

## What it does

- **Full-Text Search & Corpus Scope Filtering (FTS5):** Search across the deduplicated corpus with scope and genre filters (`All Unique Archive`, `Canonical Blog Posts`, `TypePad Era (2005–Present)`, `Pre-2005 Archive`, or `All Raw Records`), cleaned minute-level timestamps, a formatted HTML reader, and pre-verified Wayback Machine CDX snapshot links.
- **Hybrid SQL & Semantic Search / Brain:** Longitudinal research agent (`brain_engine.py`, indexed via `build_brain_index.py`) combining read-only SQLite aggregations with 768-dimensional dense vector search (`nomic-embed-text`) and local LLM reasoning (`qwen2.5:14b` via Ollama). The Brain index artifacts (`brain_index.db`, `brain_vectors.npy`) are hosted independently in the [`brain-v1.0`](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive/releases/tag/brain-v1.0) GitHub Release and downloaded on demand in the app.
- **Posting Timeline:** Volume of posts by year/month over the deduplicated corpus, eliminating inflated counts from duplicate backup folders (`copy` directories, `July 1 2003` snapshots) and aggregator index pages.
- **Asset Recovery:** Every missing image path is classified as `chart` / `chrome` / `desktop` / `other`, and recoverable figures are queried against the Wayback Machine so they can be rendered back into the article view.

## Data snapshot (deployed V3.2 database & `brain-v1.0` index, updated 2026-11)

- **27,335 total records** in the `posts` table (up from 21,531 in V2.1).
- **TypePad era added in V3.2:** 5,773 canonical TypePad posts merged (plus 31 notes/drafts and 7 skipped overlapping duplicates), bringing all-period canonical blog posts to **12,306**. The merge also added 2,258 TypePad image assets pending recovery.
- **Genre classification & Brain semantic scope (`brain-v1.0`):** Deduplicated records are classified into `essay_or_column`, `quote_commentary`, `teaching_or_meta`, `link_roundup`, and `clipping_or_mail` (excluding `archive_dump` and `corrupt_binary`). The Brain semantic index covers **14,103** deduplicated core writings (`is_unique_corpus = 1` across `essay_or_column`, `quote_commentary`, and `teaching_or_meta`).
- **LLM cleaning in V3.2:** Corrupted / garbled entries were reviewed and restored with a local `qwen2.5:7b` model; see `scripts/v3_cleaning/` for the scripts and audit logs used.
- **V2.1 baseline (pre-2005 only):** 21,531 total records, deduplicated to 11,772 unique documents (`is_unique_corpus = 1`, Jan 01, 1990 – Dec 21, 2005):
  - **6,533 canonical blog posts (2001–2005):** Verified individual Movable Type blog entries (`is_canonical_post = 1`).
  - **5,239 early static pages & course materials (1996–2000):** Deduplicated op-eds, working essays, and UC Berkeley lecture notes.
- **Verified historical provenance (V2.1 base):** **6,513** pre-verified Wayback Machine CDX snapshot URLs and **10,587** resolved canonical permalinks.
- **889 missing-asset references (pre-2005 base):** 379 DeLong charts, 195 third-party chrome, 65 desktop-folder files, 250 other — **345** snapshots found via Wayback (a 54.8% recovery rate over `chart` + `other` references).

## Run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app automatically downloads the corpus database (`weblog_corpus.db`, [`V3.2`](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive/releases/tag/V3.2)) from GitHub Releases.

To run the **Brain** longitudinal research agent locally, ensure [Ollama](https://ollama.com/) is running with the required models pulled, then download the `brain-v1.0` index artifacts (`brain_index.db` and `brain_vectors.npy`) via the in-app button in the **🧠 Brain** tab (or directly from the [`brain-v1.0`](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive/releases/tag/brain-v1.0) Release):

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

The database and Brain artifacts are not stored in this Git repository (see `.gitignore`).

## Repository layout

```text
app.py                          # Streamlit app (FTS5 search, Brain agent, timeline, asset recovery)
brain_engine.py                 # Hybrid SQL + semantic RAG engine (Ollama qwen2.5:14b + nomic-embed-text)
build_brain_index.py            # Build the Brain vector index and chunk metadata DB from weblog_corpus.db
requirements.txt                # Third-party dependencies (streamlit, pandas, numpy)
.gitignore                      # Python, virtualenvs, and downloaded database / Brain artifacts
LICENSE                         # MIT License for the code in this repository
scripts/
  classify_missing.py           # Build missing_assets table and classify missing paths (chart / chrome / desktop / other)
  wayback_recover.py            # Query the Wayback Availability API for pending missing assets
  v3_cleaning/
    merge_typepad_into_corpus.ipynb   # Merge TypePad export into weblog_corpus.db (SHA-256 dedup, rebuild FTS5)
    fast_llm_clean.py           # Clean corrupted entries with local qwen2.5:7b (writes an audit log)
    resume_llm_clean.py         # Resume an interrupted LLM cleaning run
    review_ambiguous_ollama.py / .ipynb  # Human review of ambiguous entries flagged for the LLM
    llm_clean_corpus.ipynb      # Notebook version of the LLM cleaning run
notebooks/nicole/               # Nicole Shimamoto: TypePad analysis notebooks, Wayback comparison results, poster figures
docs/emily/                     # Emily Liang: Movable Type work-to-date record and URL metadata table notes
```

## Scripts

The scripts expect `weblog_corpus.db` in the working directory (the same file the app downloads on first run).

```bash
# 1. Classify missing assets from the posts table into missing_assets
python scripts/classify_missing.py

# 2. Query Wayback for everything still pending (rate-limited and resumable)
python scripts/wayback_recover.py

# Options: --limit N to cap the run, --sleep SECONDS between requests,
# --retry-errors to also retry rows in an error state

# 3. V3.2 cleaning: merge the TypePad export, then clean corrupted entries
# (run from the directory containing weblog_corpus.db and typepad_archives.txt)
# See scripts/v3_cleaning/merge_typepad_into_corpus.ipynb
python scripts/v3_cleaning/fast_llm_clean.py

# 4. Build the Brain semantic-search index (needs local Ollama running)
python build_brain_index.py
```

## Contributors & provenance

- **Nicole Shimamoto** — TypePad-era analysis: parsed the TypePad export (`typepad_archives.txt`), compared entries against Wayback captures (`notebooks/nicole/archive_analysis2.ipynb`, `comparison_results.csv`), and produced the poster figures in `notebooks/nicole/poster_figs/`.
- **Emily Liang** — Movable Type inventories (2001–2005): chronological HTML inventories, the combined 6,533-post SQLite database, and the URL-matching metadata table (6,533 rows, Wayback found for 6,513); see `docs/emily/DeLong_archive_project_work_to_date.txt`.
- **Jason Zeng** — Corpus database, Streamlit app, V3.2 TypePad merge and LLM cleaning, and the Brain hybrid SQL + semantic retrieval layer.

Raw exports and binary artifacts (e.g. `typepad_archives.txt`, the inventory ZIPs, `weblog_corpus.db`, `brain_index.db`, `brain_vectors.npy`) are intentionally excluded from version control; the notebooks, scripts, and records above document how they were built and verified.

## License

The code in this repository is licensed under the MIT License — see `LICENSE`. The DeLong weblog text, images, and corpus database are not covered by this license.

## Data provenance and limits

- Post text, metadata, and images come from DeLong's weblog backups, curated 2001–2005 Movable Type metadata, and the Internet Archive Wayback Machine; recovered figures and snapshots are linked to Wayback URLs rather than re-hosted here.
- Coverage spans 1990–present (pre-2005 Movable Type plus the TypePad era from 2005), with dense dated coverage in 1995–2005 and 2017–2024 and sparse dated records in 2006–2016. Deduplication combines curated Movable Type canonical post matching (`6,533` posts) with rule-based filtering of backup snapshots and index pages (`5,239` early static/course pages), plus SHA-256 dedup for the TypePad merge.

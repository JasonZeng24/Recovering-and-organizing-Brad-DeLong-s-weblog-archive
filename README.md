# Brad DeLong's Weblog Archive

Recovering and organizing Brad DeLong's weblog archive — a Streamlit app built in UC Berkeley's CDSS Data Discovery practicum, covering the pre-2005 Movable Type archive and the TypePad era (2005–present).

**Live demo:** https://dsdiscovery-pre2005demo.streamlit.app/

## The problem

DeLong's Semi-Daily Journal spans decades, but surviving pre-2005 copies are scattered across overlapping backup folders, aggregator index pages, and early static course directories, while many referenced figures were never bundled locally. This project consolidates the archive into a deduplicated, searchable database with verified Wayback Machine provenance and missing-asset recovery.

## What it does

- **Full-Text Search & Corpus Scope Filtering (FTS5):** Search across the deduplicated corpus with scope filters (`Unique Archive`, `Canonical Blog Posts (2001–2005)`, `Early Pages & Courses (1996–2000)`, `TypePad Era (2005–Present)`, or `All Raw Files`), cleaned minute-level timestamps, a formatted HTML reader, and pre-verified Wayback Machine CDX snapshot links.
- **Semantic Search / DeLong Brain:** Embeddings-based concept search over the archive (`brain_engine.py`, built by `build_brain_index.py`), using local Ollama models (qwen2.5:14b for generation, nomic-embed-text for embeddings). The Brain index and vectors are downloaded from the `brain-v1.0` GitHub Release on first run and are not stored in this repository.
- **Posting Timeline:** Volume of posts by year/month over the deduplicated corpus, eliminating inflated counts from duplicate backup folders (`copy` directories, `July 1 2003` snapshots) and aggregator index pages.
- **Asset Recovery:** Every missing image path is classified as `chart` / `chrome` / `desktop` / `other`, and recoverable figures are queried against the Wayback Machine so they can be rendered back into the article view.

## Data snapshot (deployed V3.2 database, updated 2026-11)

- **27,335 total records** in the `posts` table (up from 21,531 in V2.1).
- **TypePad era added in V3.2:** 5,773 canonical TypePad posts merged (plus 31 notes/drafts and 7 skipped overlapping duplicates), bringing all-period canonical blog posts to **12,306**. The merge also added 2,258 TypePad image assets pending recovery.
- **LLM cleaning in V3.2:** Corrupted / garbled entries were reviewed and restored with a local qwen2.5:7b model; see `scripts/v3_cleaning/` for the scripts and audit logs used.
- **V2.1 baseline (pre-2005 only):** 21,531 total records, deduplicated to 11,772 unique documents (`is_unique_corpus = 1`, Jan 01, 1990 – Dec 21, 2005):
  - **6,533 canonical blog posts (2001–2005):** Verified individual Movable Type blog entries (`is_canonical_post = 1`).
  - **5,239 early static pages & course materials (1996–2000):** Deduplicated op-eds, working essays, and UC Berkeley lecture notes.
- **Verified historical provenance (V2.1 base):** **6,513** pre-verified Wayback Machine CDX snapshot URLs and **10,587** resolved canonical permalinks.
- **889 missing-asset references:** 379 DeLong charts, 195 third-party chrome, 65 desktop-folder files, 250 other.
- **Wayback image recovery:** 345 snapshots found, 188 with no snapshot, 0 pending — a 54.8% recovery rate over `chart` + `other` references (345 / 629).

The V2.1 numbers above describe the pre-2005 base release, not the current V3.2 total.

## Run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app downloads the compacted corpus database (`weblog_corpus.db`, `V3.2`) and the Brain artifacts (`brain_index.db`, `brain_vectors.npy`, `brain-v1.0`) from GitHub Releases if they are not already present. The database and Brain artifacts themselves are not stored in this repository (see `.gitignore`).

## Repository layout

```
app.py                          # Streamlit app (FTS5 search, corpus scope filters, timeline, asset recovery, Brain)
brain_engine.py                 # Semantic search / RAG engine (Ollama qwen2.5:14b + nomic-embed-text)
build_brain_index.py            # Build the Brain vector index from the corpus database
requirements.txt                # Third-party dependencies (streamlit, pandas, numpy)
.gitignore                      # Python, virtualenvs, and the downloaded database / Brain artifacts
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
- **Jason Zeng** — Corpus database, Streamlit app, V3.2 TypePad merge and LLM cleaning, and the Brain semantic-search layer.

Raw exports and databases (e.g. `typepad_archives.txt`, the inventory ZIPs, `weblog_corpus.db`, `brain_index.db`, `brain_vectors.npy`) are intentionally not committed; the notebooks, scripts, and records above document how they were used.

## License

The code in this repository is licensed under the MIT License — see `LICENSE`. The DeLong weblog text, images, and corpus database are not covered by this license.

## Data provenance and limits

- Post text, metadata, and images come from DeLong's weblog backups, curated 2001–2005 Movable Type metadata, and the Internet Archive Wayback Machine; recovered figures and snapshots are linked to Wayback URLs rather than re-hosted here.
- Coverage is 1990–present (pre-2005 Movable Type plus the TypePad era from 2005). Deduplication combines curated Movable Type canonical post matching (`6,533` posts) with rule-based filtering of backup snapshots and index pages (`5,239` early static/course pages), plus SHA-256 dedup for the TypePad merge.

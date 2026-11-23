# Brad DeLong's Weblog Archive & Brain

Recovering and organizing J. Bradford DeLong's weblog archive — a Streamlit research platform built in UC Berkeley's CDSS Data Science Discovery practicum, spanning the pre-2005 Movable Type archive and the TypePad era (2005–present).

**Live demo:** https://dsdiscovery-pre2005demo.streamlit.app/

## The problem

DeLong's *Semi-Daily Journal* spans nearly three decades, but surviving pre-2005 copies are scattered across overlapping backup folders, aggregator index pages, and early static course directories, while many referenced figures were never bundled locally. This project consolidates the archive into a deduplicated, genre-classified database with verified Wayback Machine provenance, missing-asset recovery, and a longitudinal hybrid RAG agent (**DeLong Brain**).

## What it does

- **Full-Text Search & Genre/Era Filtering (FTS5):** Search across the deduplicated corpus with scope and genre filters (`DeLong Core Voice`, `Original Essays & Columns`, `Quote + Commentary`, `Teaching & Meta`, `Clippings & Reader Mail`, `Link Roundups`), cleaned minute-level timestamps, a formatted HTML reader, and pre-verified Wayback Machine CDX snapshot links.
- **DeLong Brain — Longitudinal Research Agent (`brain_engine.py`):** Hybrid retrieval agent combining read-only SQLite aggregations (`weblog_corpus.db`) with unified all-years 768-dimensional dense vector search (`nomic-embed-text`) + BM25 lexical fusion (`posts_fts`), per-year diversity sampling, and local LLM synthesis (`qwen2.5:14b` via Ollama) with inline Wayback citations. The Brain index artifacts (`brain_index.db`, `brain_vectors.npy`) are hosted in the [`brain-v1.2`](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive/releases/tag/brain-v1.2) GitHub Release and downloaded on demand in the app.
- **Posting Timeline & Trends:** Volume of posts by year or month over the deduplicated corpus, eliminating inflated counts from duplicate backup folders (`copy` directories, `July 1 2003` snapshots) and aggregator index pages.
- **Asset Recovery & Health Audit:** Every missing image path is classified as `chart` / `chrome` / `desktop` / `other`, and recoverable figures are queried against the Wayback Machine Availability API so they can be rendered directly inside the article reader.

## Data snapshot (`V3.4` corpus & `brain-v1.2` index, updated 2026-11)

- **27,335 total indexed records** in the `posts` table, deduplicated to **16,807 unique corpus items** (`is_unique_corpus = 1`).
- **Balanced Core Corpus Distribution (`V3.4` directory & genre audit):**
  - **8,092** `essay_or_column` — original essays and syndicated columns primarily written by DeLong
  - **4,507** `quote_commentary` — substantive commentaries paired with blockquotes (**12,599** combined **DeLong Core Voice** posts)
  - **2,032** `teaching_or_meta` — university syllabi, reading lists, lecture notes, problem sets, and slide decks isolated from `Teaching_Folder/`, `macro_online/`, `refs/`, `2007_ppt/`, and `ppt/`
  - **1,590** `clipping_or_mail` — external press clippings and reader mail
  - **586** `link_roundup` — link roundups and brief pointers
- **TypePad & Movable Type Merge:** **12,306** canonical blog posts across all eras (`is_canonical_post = 1`), combining **6,533** verified pre-2005 Movable Type entries and **5,773** TypePad-era posts.
- **Semantic Index (`brain-v1.2`):** **21,151** chunked passages synchronized with `V3.4` genre classifications and precomputed `nomic-embed-text` dense embeddings (`float32`), enabling single-pass cross-decade retrieval with automatic Wayback URL injection.
- **Verified historical provenance:** **6,513** pre-verified Wayback Machine CDX snapshot URLs and **10,587** resolved canonical permalinks (pre-2005 base).
- **889 missing-asset references (pre-2005 base):** 379 DeLong charts, 195 third-party chrome, 65 desktop-folder files, 250 other — **345** snapshots recovered via Wayback (a 54.8% recovery rate over `chart` + `other` references), plus 2,258 TypePad image references triaged for recovery.

## Run

Requires Python 3.10+.

```bash
git clone [https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive.git](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive.git)
cd Recovering-and-organizing-Brad-DeLong-s-weblog-archive
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app automatically downloads the corpus database (`weblog_corpus.db`, [`V3.4`](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive/releases/tag/V3.4)) from GitHub Releases.

To run the **DeLong Brain** longitudinal research agent locally, ensure [Ollama](https://ollama.com/) is running with the required models pulled, then download the `brain-v1.2` index artifacts (`brain_index.db` and `brain_vectors.npy`) via the in-app button in the **🧠 Brain** tab (or directly from the [`brain-v1.2`](https://github.com/JasonZeng24/Recovering-and-organizing-Brad-DeLong-s-weblog-archive/releases/tag/brain-v1.2) Release):

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

The binary database and Brain vector artifacts are excluded from Git version control (see `.gitignore`).

## Repository layout

```text
app.py                          # Streamlit app (FTS5 search, DeLong Brain agent, timeline, asset recovery)
brain_engine.py                 # Hybrid SQL + all-years semantic RAG engine (Ollama qwen2.5:14b + nomic-embed-text)
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

# 3. Corpus cleaning: merge the TypePad export, then clean corrupted entries
# (run from the directory containing weblog_corpus.db and typepad_archives.txt)
# See scripts/v3_cleaning/merge_typepad_into_corpus.ipynb
python scripts/v3_cleaning/fast_llm_clean.py

# 4. Build the Brain semantic-search index (needs local Ollama running)
python build_brain_index.py
```

## Contributors & provenance

- **Nicole Shimamoto** — TypePad-era analysis: parsed the TypePad export (`typepad_archives.txt`), compared entries against Wayback captures (`notebooks/nicole/archive_analysis2.ipynb`, `comparison_results.csv`), and produced the poster figures in `notebooks/nicole/poster_figs/`.
- **Emily Liang** — Movable Type inventories (2001–2005): chronological HTML inventories, the combined 6,533-post SQLite database, and the URL-matching metadata table (6,533 rows, Wayback found for 6,513); see `docs/emily/DeLong_archive_project_work_to_date.txt`.
- **Jason Zeng** — Corpus database, Streamlit app, TypePad merge, LLM cleaning & V3.4 genre/directory audit, and the DeLong Brain hybrid SQL + semantic retrieval engine.

Raw exports and binary artifacts (e.g. `typepad_archives.txt`, the inventory ZIPs, `weblog_corpus.db`, `brain_index.db`, `brain_vectors.npy`) are intentionally excluded from version control; the notebooks, scripts, and records above document how they were built and verified.

## License

The code in this repository is licensed under the MIT License — see `LICENSE`. The DeLong weblog text, images, and corpus database are not covered by this license.

## Data provenance and limits

- Post text, metadata, and images come from DeLong's weblog backups, curated 2001–2005 Movable Type metadata, and the Internet Archive Wayback Machine; recovered figures and snapshots are linked to Wayback URLs rather than re-hosted here.
- Coverage spans 1990–present (pre-2005 Movable Type plus the TypePad era from 2005), with dense dated coverage in 1995–2005 and 2017–2024 and sparse dated records in 2006–2016. Deduplication combines curated Movable Type canonical post matching (`6,533` posts) with rule-based filtering of backup snapshots and index pages, directory-based syllabus/course cleanup (`2,032` `teaching_or_meta` items), and SHA-256 dedup for the TypePad merge.

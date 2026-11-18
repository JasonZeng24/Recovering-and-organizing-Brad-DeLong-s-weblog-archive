import json
import os
import re
import sqlite3
import time
import urllib.request
import numpy as np

SRC_DB = "weblog_corpus.db"
INDEX_DB = "brain_index.db"
VEC_PATH = "brain_vectors.npy"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 64


def chunk_text(text: str, max_words: int = 380, overlap: int = 60):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur_words, cur_len = [], [], 0
    for p in paragraphs:
        w = p.split()
        if cur_len + len(w) > max_words and cur_words:
            chunks.append(" ".join(cur_words))
            cur_words = cur_words[-overlap:] if overlap > 0 else []
            cur_len = len(cur_words)
        cur_words.extend(w)
        cur_len += len(w)
    if cur_words:
        chunks.append(" ".join(cur_words))
    return chunks


def get_embeddings(texts):
    prefixed = [f"search_document: {t}" for t in texts]
    payload = {"model": EMBED_MODEL, "input": prefixed, "keep_alive": "20m"}
    req = urllib.request.Request(
        EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        arr = np.array(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / np.maximum(norms, 1e-12)).astype(np.float16)


def main():
    src = sqlite3.connect(SRC_DB)
    rows = src.execute(
        """
        SELECT id, title, exact_post_datetime, corpus_era, post_genre, wayback_url, body_clean
        FROM posts
        WHERE is_unique_corpus = 1
          AND post_genre IN ('essay_or_column', 'quote_commentary', 'teaching_or_meta')
          AND length(trim(body_clean)) > 80
        ORDER BY exact_post_datetime ASC, id ASC
        """
    ).fetchall()
    src.close()

    if os.path.exists(INDEX_DB):
        os.remove(INDEX_DB)
    dst = sqlite3.connect(INDEX_DB)
    dst.execute(
        """
        CREATE TABLE chunks (
            chunk_idx INTEGER PRIMARY KEY,
            post_id INTEGER,
            year INTEGER,
            post_date TEXT,
            corpus_era TEXT,
            post_genre TEXT,
            title TEXT,
            wayback_url TEXT,
            chunk_text TEXT
        )
        """
    )

    all_chunks = []
    for pid, title, dt, era, genre, wb_url, body in rows:
        dt_str = (dt or "")[:10]
        year = int(dt_str[:4]) if len(dt_str) >= 4 and dt_str[:4].isdigit() else 0
        pieces = chunk_text(body)
        for piece in pieces:
            header = f"[{dt_str or era} | {genre}] {title or 'Untitled'}"
            all_chunks.append((
                len(all_chunks), pid, year, dt_str, era, genre,
                title or "Untitled", wb_url or "", f"{header}\n{piece}"
            ))

    print(f"核心文章 {len(rows):,} 篇已切分为 {len(all_chunks):,} 个英文语义块，开始向量化...", flush=True)
    dst.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", all_chunks)
    dst.commit()
    dst.close()

    vectors = []
    t0 = time.time()
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_texts = [c[8] for c in all_chunks[i : i + BATCH_SIZE]]
        vecs = get_embeddings(batch_texts)
        vectors.append(vecs)
        done = min(i + BATCH_SIZE, len(all_chunks))
        if done % (BATCH_SIZE * 20) == 0 or done == len(all_chunks):
            rate = done / max(time.time() - t0, 0.1)
            eta = (len(all_chunks) - done) / max(rate, 0.1)
            print(f"  向量化进度: {done:,}/{len(all_chunks):,} | 速度: {rate:.1f} 块/秒 | 预计剩余: {eta:.0f}s", flush=True)

    mat = np.vstack(vectors)
    np.save(VEC_PATH, mat)
    print(f"索引构建完成！向量矩阵维度: {mat.shape}，已保存至 {VEC_PATH} 与 {INDEX_DB}")


if __name__ == "__main__":
    main()

import json
import os
import sqlite3
import time
import urllib.request

DB_PATH = "weblog_corpus.db"
AMBIGUOUS_IN = "ambiguous_for_llm.jsonl"
CHECKPOINT_OUT = "ollama_reviewed.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

VALID_GENRES = {
    "essay_or_column",
    "quote_commentary",
    "clipping_or_mail",
    "link_roundup",
    "teaching_or_meta",
}

SYSTEM_PROMPT = """You are a corpus classifier for economist J. Bradford DeLong's blog.
Classify each post into EXACTLY ONE of these 5 genres based on its title, opening, and closing:
- "essay_or_column": Original essay, economic analysis, or syndicated column written primarily by DeLong himself (quotes < 30%).
- "quote_commentary": Excerpts/quotes from another author (30%-80%) paired with substantive commentary, critique, or rebuttal by DeLong.
- "clipping_or_mail": Pure bookmark/excerpt from another writer, reader email ("From the inbox"), or guest post with minimal/no commentary by DeLong (e.g., ending with "Read the whole thing", "Indeed", or "#noted").
- "link_roundup": List of multiple external links, daily reading dumps, or tweet roundups.
- "teaching_or_meta": Course syllabus, reading list, lecture notes, problem set, or blog housekeeping notice.

Respond ONLY with valid JSON: {"genre": "<one_of_the_5_labels>"}"""


def classify_with_ollama(rec):
    user_prompt = (
        f"Title: {rec.get('title', '')}\n"
        f"Categories: {rec.get('categories', '')}\n"
        f"--- OPENING ---\n{rec.get('head', '')}\n"
        f"--- CLOSING ---\n{rec.get('tail', '')}"
    )
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 32},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        parsed = json.loads(res.get("response", "{}"))
        genre = str(parsed.get("genre", "")).strip()
        return genre if genre in VALID_GENRES else None


def main():
    if not os.path.exists(AMBIGUOUS_IN):
        raise FileNotFoundError(f"找不到 {AMBIGUOUS_IN}")

    done = {}
    if os.path.exists(CHECKPOINT_OUT):
        with open(CHECKPOINT_OUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done[obj["id"]] = obj["genre"]
        print(f"已加载历史进度：{len(done):,} 条已完成", flush=True)

    with open(AMBIGUOUS_IN, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    todo = [r for r in records if r["id"] not in done]
    print(f"待复核样本：{len(todo):,} 条（使用模型: {MODEL_NAME}）...", flush=True)

    t0 = time.time()
    errors = 0
    with open(CHECKPOINT_OUT, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(todo, 1):
            try:
                genre = classify_with_ollama(rec)
                if genre:
                    done[rec["id"]] = genre
                    out_f.write(json.dumps({"id": rec["id"], "genre": genre}) + "\n")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [警告] ID {rec['id']} 请求失败: {e}", flush=True)

            if i % 100 == 0 or i == len(todo):
                out_f.flush()
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.1)
                eta = (len(todo) - i) / max(rate, 0.1)
                print(
                    f"  进度: {i:,}/{len(todo):,} ({i/len(todo):.1%}) | "
                    f"速度: {rate:.1f} 条/秒 | 预计剩余: {eta:.0f}s",
                    flush=True,
                )

    print("\n正在将复核结果写回 weblog_corpus.db ...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    updates = [(g, pid) for pid, g in done.items()]
    cur.executemany(
        "UPDATE posts SET post_genre = ?, needs_llm_review = 0 WHERE id = ?",
        updates,
    )
    conn.commit()

    print("\n=== 最终去重语料（17,526 篇）体裁分布 ===")
    for genre, cnt in cur.execute("""
        SELECT post_genre, COUNT(*) AS c
        FROM posts
        WHERE is_unique_corpus = 1
        GROUP BY post_genre
        ORDER BY c DESC
    """):
        print(f"  {genre:<20}: {cnt:,} 篇")
    conn.close()


if __name__ == "__main__":
    main()

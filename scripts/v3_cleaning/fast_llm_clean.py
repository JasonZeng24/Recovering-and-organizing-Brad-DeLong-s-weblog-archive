import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import time
import urllib.request

DB_PATH = "weblog_corpus.db"
AUDIT_LOG = "llm_clean_audit.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
MAX_WORKERS = 1

DUMP_FILENAME_RE = re.compile(r"^(exported[-_]|index\.html?|archives?\.html?|all_posts|default\.html?)", re.I)
SUSPECT_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]|"
    r"Tranfer path:|Transfer path:|Subject:\s*Homework|"
    r"[^\w\s.,;:'\"!?()\[\]{}\-–—/\\@#$%&*+=<>|~`^‘’“”…£€¥¢§°±×÷]{3,}"
)

SYSTEM_PROMPT = """You are a strict archival text restorer for economist J. Bradford DeLong's blog.
Clean corrupted text while preserving 100% of authentic human-written prose VERBATIM.
RULES:
1. If the entire input is unreadable binary garbage, random symbols, or spam headers with no real blog prose, output ONLY:
[[PURE_GARBAGE]]
2. If real prose is mixed with garbled symbols or spam headers, remove ONLY the noise, keep ALL valid prose WORD-FOR-WORD, and wrap the result inside tags:
<CLEAN_TEXT>
the verbatim cleaned text here
</CLEAN_TEXT>
Do NOT output JSON. Do NOT add commentary."""


def is_suspect_doc(body: str) -> bool:
    if not body or len(body.strip()) < 25:
        return True
    if SUSPECT_RE.search(body):
        return True
    no_url = re.sub(r"https?://\S+", "", body)
    words = no_url.split()
    return bool(words and max(len(w) for w in words) > 55)


def clean_chunk(text_chunk: str):
    text_chunk = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]", " ", text_chunk)
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": f"Clean the following archival text:\n\n{text_chunk}",
        "stream": False,
        "keep_alive": "20m",
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 2048},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw = res.get("response", "").strip()
        if "[[PURE_GARBAGE]]" in raw:
            return "PURE_GARBAGE", ""
        m = re.search(r"<CLEAN_TEXT>\s*(.*?)\s*</CLEAN_TEXT>", raw, re.S)
        if m:
            return "REPAIRED", m.group(1).strip()
        return "REPAIRED", raw


def process_record(rec):
    pid, rel_path, body = rec
    paragraphs = re.split(r"\n\s*\n", body)
    if len(body) > 4500 and len(paragraphs) > 1:
        rebuilt, any_repaired, llm_calls = [], False, 0
        for p in paragraphs:
            if is_suspect_doc(p) and llm_calls < 4:
                llm_calls += 1
                st, cleaned = clean_chunk(p[:3500])
                if st == "REPAIRED" and cleaned:
                    rebuilt.append(cleaned)
                    any_repaired = True
                elif st == "PURE_GARBAGE":
                    any_repaired = True
                else:
                    rebuilt.append(p)
            else:
                rebuilt.append(p)
        final_text = "\n\n".join(rebuilt).strip()
        if len(final_text.split()) < 25:
            return pid, rel_path, "PURE_GARBAGE", body, ""
        return pid, rel_path, ("REPAIRED" if any_repaired else "UNCHANGED"), body, final_text

    st, cleaned = clean_chunk(body[:4000])
    if st == "PURE_GARBAGE" or len(cleaned.split()) < 25:
        return pid, rel_path, "PURE_GARBAGE", body, ""
    if st == "REPAIRED" and cleaned:
        return pid, rel_path, "REPAIRED", body, cleaned
    return pid, rel_path, "UNCHANGED", body, body


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT id, rel_path, corpus_era, is_canonical_post, body_clean FROM posts").fetchall()

    dump_ids = []
    suspects = []
    for pid, rpath, era, is_canon, body in rows:
        body = body or ""
        fname = os.path.basename(rpath or "")
        # 先毫秒级隔离 Pre-2005 整月导出大杂烩，避免拖慢 LLM
        if era == "pre-2005" and is_canon == 0 and (DUMP_FILENAME_RE.search(fname) or len(body.split()) > 3500):
            dump_ids.append((pid,))
        elif is_suspect_doc(body):
            suspects.append((pid, rpath, body))

    print(f"规则隔离导出大杂烩: {len(dump_ids):,} 条 | 待 LLM 精修疑似样本: {len(suspects):,} 条", flush=True)

    repaired_updates, garbage_ids = [], []
    t0, done_cnt, lock = time.time(), 0, threading.Lock()

    with open(AUDIT_LOG, "w", encoding="utf-8") as log_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_record, r) for r in suspects]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    pid, rpath, status, orig_body, new_body = fut.result()
                    with lock:
                        done_cnt += 1
                        if status == "PURE_GARBAGE":
                            garbage_ids.append((pid,))
                            log_f.write(json.dumps({"id": pid, "rel_path": rpath, "action": "ISOLATED_GARBAGE"}, ensure_ascii=False) + "\n")
                        elif status == "REPAIRED" and new_body != orig_body:
                            repaired_updates.append((new_body, pid))
                            log_f.write(json.dumps({"id": pid, "rel_path": rpath, "action": "REPAIRED_TEXT", "before": orig_body[:200], "after": new_body[:200]}, ensure_ascii=False) + "\n")
                        if done_cnt % 15 == 0 or done_cnt == len(suspects):
                            log_f.flush()
                            rate = done_cnt / max(time.time() - t0, 0.1)
                            eta = (len(suspects) - done_cnt) / max(rate, 0.1)
                            print(f"  进度: {done_cnt}/{len(suspects)} | 速度: {rate:.2f} 条/秒 | 预计剩余: {eta:.0f}s", flush=True)
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode("utf-8", errors="ignore")[:120]
                    with lock:
                        done_cnt += 1
                        print(f"  [HTTP {e.code}] {err_body}", flush=True)
                except Exception as e:
                    with lock:
                        done_cnt += 1
                        print(f"  [跳过异常] {e}", flush=True)

    if dump_ids:
        cur.executemany("UPDATE posts SET is_unique_corpus = 0, post_genre = 'archive_dump' WHERE id = ?", dump_ids)
    if repaired_updates:
        cur.executemany("UPDATE posts SET body_clean = ? WHERE id = ?", repaired_updates)
    if garbage_ids:
        cur.executemany("UPDATE posts SET is_unique_corpus = 0, is_canonical_post = 0, post_genre = 'corrupt_binary' WHERE id = ?", garbage_ids)

    print("正在重建 FTS5 索引...", flush=True)
    cur.execute("INSERT INTO posts_fts(posts_fts) VALUES('rebuild');")
    conn.commit()
    conn.close()
    print(f"全部完成！隔离大杂烩: {len(dump_ids)} | 挽救正文: {len(repaired_updates)} | 隔离纯坏档: {len(garbage_ids)}")


if __name__ == "__main__":
    main()

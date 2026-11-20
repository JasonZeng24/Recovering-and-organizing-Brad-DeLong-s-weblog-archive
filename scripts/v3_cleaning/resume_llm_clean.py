import json
import sqlite3
import time
from fast_llm_clean import DB_PATH, MODEL_NAME, DUMP_FILENAME_RE, is_suspect_doc, process_record
import os

OLD_AUDIT = "llm_clean_audit.jsonl"
NEW_AUDIT = "llm_clean_audit_final.jsonl"

def main():
    prev_garbage_ids = set()
    prev_repaired_ids = set()
    if os.path.exists(OLD_AUDIT):
        with open(OLD_AUDIT, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("action") == "ISOLATED_GARBAGE":
                    prev_garbage_ids.add(rec["id"])
                elif rec.get("action") == "REPAIRED_TEXT":
                    prev_repaired_ids.add(rec["id"])

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT id, rel_path, corpus_era, is_canonical_post, body_clean FROM posts").fetchall()

    dump_ids = []
    suspects = []
    for pid, rpath, era, is_canon, body in rows:
        body = body or ""
        fname = os.path.basename(rpath or "")
        if era == "pre-2005" and is_canon == 0 and (DUMP_FILENAME_RE.search(fname) or len(body.split()) > 3500):
            dump_ids.append((pid,))
        elif is_suspect_doc(body):
            suspects.append((pid, rpath, body))

    # 前 360 条已扫描过：跳过已确认正常的，只重跑少数需修复全文的；360 之后全部跑
    done_slice = suspects[:360]
    rem_slice = suspects[360:]
    re_run_repaired = [r for r in done_slice if r[0] in prev_repaired_ids]
    to_process = re_run_repaired + rem_slice

    print(
        f"断点恢复成功：已直接继承坏档 {len(prev_garbage_ids)} 条，跳过前 360 条正常文章；"
        f"本次仅需处理 {len(to_process)} 条（含 {len(re_run_repaired)} 条修复复原 + {len(rem_slice)} 条未完成）",
        flush=True,
    )

    repaired_updates = []
    garbage_ids = [(pid,) for pid in prev_garbage_ids]
    t0 = time.time()

    with open(NEW_AUDIT, "w", encoding="utf-8") as log_f:
        for pid in prev_garbage_ids:
            log_f.write(json.dumps({"id": pid, "action": "ISOLATED_GARBAGE_RESUMED"}, ensure_ascii=False) + "\n")

        for idx, rec in enumerate(to_process, 1):
            try:
                pid, rpath, status, orig_body, new_body = process_record(rec)
                if status == "PURE_GARBAGE":
                    garbage_ids.append((pid,))
                    log_f.write(json.dumps({"id": pid, "rel_path": rpath, "action": "ISOLATED_GARBAGE"}, ensure_ascii=False) + "\n")
                elif status == "REPAIRED" and new_body != orig_body:
                    repaired_updates.append((new_body, pid))
                    log_f.write(json.dumps({"id": pid, "rel_path": rpath, "action": "REPAIRED_TEXT", "before": orig_body[:200], "after": new_body[:200]}, ensure_ascii=False) + "\n")

                if idx % 10 == 0 or idx == len(to_process):
                    log_f.flush()
                    rate = idx / max(time.time() - t0, 0.1)
                    eta = (len(to_process) - idx) / max(rate, 0.1)
                    print(f"  续跑进度: {idx}/{len(to_process)} | 速度: {rate:.2f} 条/秒 | 预计剩余: {eta:.0f}s", flush=True)
            except Exception as e:
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

import os
import argparse
import sqlite3
import time
import pandas as pd
from tqdm import tqdm
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from generate_dataset import EMOTION_LABELS, normalize_label


def setup_cache(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_labels (
            phrase TEXT PRIMARY KEY,
            emotion TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_verified_emotion ON verified_labels(emotion);")
    conn.commit()
    return conn


def cache_get_bulk(conn: sqlite3.Connection, phrases):
    cur = conn.cursor()
    out = {}
    for p in phrases:
        try:
            row = cur.execute("SELECT emotion FROM verified_labels WHERE phrase=?", (p,)).fetchone()
            if row:
                out[p] = row[0]
        except Exception:
            continue
    return out


def cache_put(conn: sqlite3.Connection, phrase: str, emotion: str):
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR REPLACE INTO verified_labels(phrase, emotion) VALUES(?, ?)", (phrase, emotion))
    except Exception:
        pass


def classify_phrase_with_verification(phrase: str, current_label: str, model: str, host: str, temperature: float, timeout: int = 60) -> str:
    prompt = f"""
You are verifying the emotional label of this phrase, to make sure that it matches the defined emotional categories.
If the assigned label seems correct, return it.
If the label is incorrect, change it to the most appropriate one.

Return ONLY ONE of the following labels:
neutral, happy, sad, angry, surprised, disgusted, mischievous, love.

Use these definitions:
- neutral: neutral/calm state, phrase without strong emotion
- happy: joy, happiness, very positive emotion
- sad: sadness, melancholy, feeling down
- angry: anger, frustration, very negative emotion
- surprised: surprise, shock, fear, unexpected event
- disgusted: disgust, revulsion, aversion
- mischievous: playful, sassy, sexy, seductive
- love: love, explicit affection, romantic

Phrase: {phrase}
Assigned label: {current_label}
Label:
"""
    client = Client(host=host)
    data = client.generate(model=model, prompt=prompt, options={"temperature": temperature})
    return normalize_label(data.get("response", ""))


def classify_phrase_with_retries(phrase: str, current_label: str, model: str, host: str, temperature: float, retries: int, backoff: float) -> str:
    attempt = 0
    while True:
        try:
            return classify_phrase_with_verification(phrase, current_label, model, host, temperature)
        except Exception:
            attempt += 1
            if attempt > retries:
                return ""
            time.sleep(backoff * attempt)


def verify_dataset(df: pd.DataFrame, model: str, host: str, temperature: float, workers: int, cache_conn: sqlite3.Connection, retries: int, backoff: float, commit_interval: int = 500):
    phrases = df["text"].astype(str).tolist()
    cached = cache_get_bulk(cache_conn, phrases)
    results = {}
    to_verify = [(p, df.loc[i, "emotion"]) for i, p in enumerate(phrases) if p not in cached]
    print(f"Cached labels: {len(cached)} | To verify: {len(to_verify)}")
    cur = cache_conn.cursor()
    processed_since_commit = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(classify_phrase_with_retries, p, lab, model, host, temperature, retries, backoff): p for p, lab in to_verify}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Verifying", unit="phrase"):
            p = futs[fut]
            try:
                label = fut.result()
            except Exception:
                label = ""
            results[p] = label
            cache_put(cache_conn, p, label)
            processed_since_commit += 1
            if processed_since_commit >= commit_interval:
                cache_conn.commit()
                processed_since_commit = 0
    cache_conn.commit()
    all_results = dict(cached)
    all_results.update(results)
    updated = 0
    empty_filled = 0
    for i, row in df.iterrows():
        p = str(row["text"])
        new_label = all_results.get(p, "").strip().lower()
        old_label = str(row.get("emotion", "")).strip().lower()
        if not old_label or old_label not in EMOTION_LABELS:
            df.at[i, "emotion"] = new_label
            empty_filled += 1
        elif new_label != old_label:
            df.at[i, "emotion"] = new_label
            updated += 1
    return df, updated, empty_filled


def main():
    parser = argparse.ArgumentParser(description="Parallel label verification and improvement for emotion dataset")
    parser.add_argument("--input-csv", required=True, help="Path to the input dataset CSV (must have text and emotion columns)")
    parser.add_argument("--output-csv", default="./data/emotions_improved.csv", help="Output path for improved dataset")
    parser.add_argument("--cache-db", default="./data/improvement_cache.sqlite", help="SQLite cache DB path")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server host")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    cache_conn = setup_cache(args.cache_db)
    df_out, updated, filled = verify_dataset(df=df, model=args.model, host=args.host, temperature=args.temperature, workers=args.workers, cache_conn=cache_conn, retries=args.retries, backoff=args.retry_backoff)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"\nSaved improved dataset to: {args.output_csv}")
    print(f"Updated incorrect labels: {updated}")
    print(f"Filled empty labels: {filled}")
    print("Label distribution after improvement:")
    print(df_out["emotion"].value_counts())


if __name__ == "__main__":
    main()

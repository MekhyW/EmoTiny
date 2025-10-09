"""
Generate a labeled emotion dataset from raw texts using a local LLM via Ollama.

Reads all text files from a data directory, splits them into phrases, classifies each phrase into one
of the 8 EmoTiny labels using Ollama, and saves the resulting dataset to Parquet and optionally CSV.
"""

import os
import re
import time
import argparse
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Iterable, Optional
import pandas as pd
from ollama import Client
from tqdm import tqdm

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "mischievous", "love"]

def iter_text_file_paths(data_dir: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> Iterable[str]:
    paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in extensions and "index" not in name.lower():
                paths.append(os.path.join(root, name))
    paths.sort() # Deterministic order for consistent resume behavior
    for p in paths:
        yield p


def read_text_files(data_dir: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
    texts: List[str] = []
    for path in iter_text_file_paths(data_dir, extensions):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        except Exception:
            pass
    return texts


def split_into_phrases(text: str, min_words: int = 3, max_chars: int = 400) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text)
    phrases: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if len(s.split()) < min_words:
            continue
        if len(s) > max_chars:
            continue
        phrases.append(s)
    return phrases


def iter_phrases(data_dir: str, min_words: int = 3, max_chars: int = 400) -> Iterable[str]:
    for path in iter_text_file_paths(data_dir):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        for phrase in split_into_phrases(text, min_words=min_words, max_chars=max_chars):
            yield phrase


def normalize_label(raw: str) -> str:
    t = raw.split()[0].strip().lower().strip("'\"` .!?")
    mapping: Dict[str, str] = {
        "neutral": "neutral",
        "calm": "neutral",
        "okay": "neutral",
        "happy": "happy",
        "joy": "happy",
        "joyful": "happy",
        "sad": "sad",
        "sadness": "sad",
        "angry": "angry",
        "anger": "angry",
        "mad": "angry",
        "surprised": "surprised",
        "surprise": "surprised",
        "shocked": "surprised",
        "disgusted": "disgusted",
        "disgust": "disgusted",
        "grossed": "disgusted",
        "mischievous": "mischievous",
        "mischief": "mischievous",
        "playful": "mischievous",
        "sassy": "mischievous",
        "love": "love",
        "affection": "love",
        # Portuguese common mappings
        "neutro": "neutral",
        "feliz": "happy",
        "triste": "sad",
        "raiva": "angry",
        "irritado": "angry",
        "surpreso": "surprised",
        "nojo": "disgusted",
        "nojento": "disgusted",
        "malicioso": "mischievous",
        "safado": "mischievous",
        "amor": "love",
        "apaixonado": "love",
    }
    canonical = mapping.get(t, t)
    if canonical in EMOTION_LABELS:
        return canonical
    return "neutral"


def classify_phrase_with_ollama(phrase: str, model: str = "gemma3:1b", host: str = "http://localhost:11434", temperature: float = 0.0, timeout: int = 60) -> str:
    prompt = f"""
You are an emotion classifier for short text in English and Portuguese.

Classify the emotion of the following phrase into EXACTLY ONE of these labels: neutral, happy, sad, angry, surprised, disgusted, mischievous, love.

Use these definitions:
- neutral: neutral/calm state, phrase without strong emotion
- happy: joy, happiness, very positive emotion
- sad: sadness, melancholy, feeling down
- angry: anger, frustration, very negative emotion
- surprised: surprise, shock, fear, unexpected event
- disgusted: disgust, revulsion, aversion
- mischievous: playful, sassy, sexy, seductive
- love: love, explicit affection, romantic

Return ONLY the label word.
Do NOT include any punctuation, and do NOT justify your choice.
Use English labels even for Portuguese text.

Phrase: {phrase}
Label:
    """
    client = Client(host=host)
    data = client.generate(model=model, prompt=prompt, options={"temperature": temperature})
    return normalize_label(data.get("response", ""))


def classify_phrase_with_retries(phrase: str, model: str, host: str, temperature: float, timeout: int, retries: int = 3, backoff: float = 1.5,) -> str:
    attempt = 0
    while True:
        try:
            return classify_phrase_with_ollama(phrase, model=model, host=host, temperature=temperature, timeout=timeout)
        except Exception:
            attempt += 1
            if attempt > retries:
                return "neutral"
            time.sleep(backoff * attempt)


def ensure_checkpoint_dirs(base_dir: str) -> Tuple[str, Optional[str]]:
    parquet_dir = os.path.join(base_dir, "parquet")
    csv_dir = os.path.join(base_dir, "csv")
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    index_path = os.path.join(base_dir, "index.txt") # Index file to track written chunks and row counts
    if not os.path.exists(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("# part_path rows\n")
    return parquet_dir, csv_dir


def get_resume_state(base_dir: str) -> Tuple[int, int]:
    """Return (next_part_id, total_rows_written) using index.txt in checkpoint dir."""
    index_path = os.path.join(base_dir, "index.txt")
    next_part_id = 1
    total_rows = 0
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    _, rows_str = line.split()
                    total_rows += int(rows_str)
                    next_part_id += 1
                except Exception:
                    continue
    return next_part_id, total_rows


def append_index(base_dir: str, part_path: str, rows: int) -> None:
    index_path = os.path.join(base_dir, "index.txt")
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(f"{part_path} {rows}\n")


def chunk_list(lst: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def setup_cache(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            phrase TEXT PRIMARY KEY,
            emotion TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_labels_emotion ON labels(emotion);")
    return conn


def cache_get_bulk(conn: sqlite3.Connection, phrases: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    cur = conn.cursor()
    for p in phrases:
        try:
            row = cur.execute("SELECT emotion FROM labels WHERE phrase=?", (p,)).fetchone()
            if row:
                out[p] = row[0]
        except Exception:
            continue
    return out


def cache_put_bulk(conn: sqlite3.Connection, items: List[Tuple[str, str]]) -> None:
    cur = conn.cursor()
    try:
        cur.executemany("INSERT OR REPLACE INTO labels(phrase, emotion) VALUES(?, ?)", items)
    except Exception:
        pass # best effort; ignore


def main():
    parser = argparse.ArgumentParser(description="Generate emotion dataset using a small LLM via Ollama (fast + resumable)")
    parser.add_argument("--data-dir", default="./data", help="Directory containing raw text files")
    parser.add_argument("--output-parquet", default="./data/emotions.parquet", help="Optional final merged Parquet path")
    parser.add_argument("--output-csv", default="", help="Optional final merged CSV path")
    parser.add_argument("--checkpoint-dir", default="", help="Directory to store chunked checkpoints (parquet/csv + index)")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server host")
    parser.add_argument("--min-words", type=int, default=2, help="Minimum words per phrase")
    parser.add_argument("--max-chars", type=int, default=400, help="Maximum characters per phrase")
    parser.add_argument("--max-phrases", type=int, default=0, help="Maximum number of phrases to label (0 = no limit)")
    parser.add_argument("--workers", type=int, default=100, help="Number of concurrent workers for classification")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Number of phrases per checkpoint chunk")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint index if present")
    parser.add_argument("--unique", action="store_true", help="Skip writing duplicate phrases, using cache for detection")
    parser.add_argument("--cache-db", default="./data/emotions_cache.sqlite", help="SQLite cache DB path for phrase→label")
    parser.add_argument("--retries", type=int, default=3, help="Classification retries on failure")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Backoff multiplier between retries (seconds)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--timeout", type=int, default=60, help="LLM request timeout seconds")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir:
        base = os.path.splitext(os.path.basename(args.output_parquet))[0] or "emotions"
        checkpoint_dir = os.path.join(os.path.dirname(args.output_parquet), f"{base}_chunks")
    os.makedirs(checkpoint_dir, exist_ok=True)
    parquet_dir, csv_dir = ensure_checkpoint_dirs(checkpoint_dir)
    cache_conn = setup_cache(args.cache_db)
    next_part_id, rows_written_so_far = get_resume_state(checkpoint_dir)
    if args.resume and rows_written_so_far > 0:
        print(f"Resume enabled. Found {rows_written_so_far} rows in checkpoints. Continuing at part {next_part_id}.")

    def write_chunk(part_id: int, phrases: List[str], emotions: List[str]):
        df = pd.DataFrame({"text": phrases, "emotion": emotions})
        part_name = f"part-{part_id:06d}"
        parquet_path = os.path.join(parquet_dir, part_name + ".parquet")
        df.to_parquet(parquet_path, index=False)
        if args.output_csv:
            csv_path = os.path.join(csv_dir, part_name + ".csv")
            df.to_csv(csv_path, index=False)
        append_index(checkpoint_dir, os.path.join("parquet", part_name + ".parquet"), len(df))

    phrases_buffer: List[str] = []
    processed_count = 0
    part_id = next_part_id
    skip_remaining = rows_written_so_far if args.resume else 0
    skipped = 0

    def maybe_emit_chunk():
        nonlocal part_id, phrases_buffer, processed_count
        if not phrases_buffer:
            return
        cached = cache_get_bulk(cache_conn, phrases_buffer)
        to_classify = [p for p in phrases_buffer if p not in cached]
        results: Dict[str, str] = dict(cached)
        if to_classify:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = {
                    ex.submit(classify_phrase_with_retries, p, args.model, args.host, args.temperature, args.timeout, args.retries, args.retry_backoff): p for p in to_classify
                }
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Classifying", unit="phrase", leave=False):
                    p = futs[fut]
                    try:
                        lab = fut.result()
                    except Exception:
                        lab = "neutral"
                    results[p] = lab
        cache_put_bulk(cache_conn, [(p, results[p]) for p in phrases_buffer]) # Persist cache for all buffer items
        if args.unique:
            pre_cached = set(cached.keys())
            phrases_to_write = [p for p in phrases_buffer if p not in pre_cached]
        else:
            phrases_to_write = list(phrases_buffer)
        emotions_to_write = [results[p] for p in phrases_to_write]
        write_chunk(part_id, phrases_to_write, emotions_to_write)
        processed_count += len(phrases_to_write)
        part_id += 1
        phrases_buffer = []

    phrase_iter = iter_phrases(args.data_dir, min_words=args.min_words, max_chars=args.max_chars)
    for phrase in phrase_iter:
        if args.max_phrases and processed_count >= args.max_phrases:
            break
        if skip_remaining > 0:
            skip_remaining -= 1
            skipped += 1
            continue
        phrases_buffer.append(phrase)
        if len(phrases_buffer) >= args.chunk_size:
            maybe_emit_chunk()
    maybe_emit_chunk()

    print(f"Processed and wrote {processed_count} new rows.")
    print(f"Checkpoints stored in: {checkpoint_dir}")

    if args.output_parquet:
        try:
            import glob
            part_files = sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))
            if part_files:
                dfs = [pd.read_parquet(pf) for pf in part_files]
                final_df = pd.concat(dfs, ignore_index=True)
                os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
                final_df.to_parquet(args.output_parquet, index=False)
                print(f"Merged final Parquet: {args.output_parquet} ({len(final_df)} rows)")
        except Exception as e:
            print(f"Skipping Parquet merge due to error: {e}")
    if args.output_csv:
        try:
            import glob
            part_files = sorted(glob.glob(os.path.join(csv_dir, "part-*.csv")))
            if part_files:
                dfs = [pd.read_csv(pf) for pf in part_files]
                final_df = pd.concat(dfs, ignore_index=True)
                os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
                final_df.to_csv(args.output_csv, index=False)
                print(f"Merged final CSV: {args.output_csv} ({len(final_df)} rows)")
        except Exception as e:
            print(f"Skipping CSV merge due to error: {e}")

    try:
        if args.output_parquet and os.path.exists(args.output_parquet): # Show simple distribution by reading merged parquet if available, else latest chunk
            df = pd.read_parquet(args.output_parquet)
            print("Dataset label distribution (final):")
            print(df["emotion"].value_counts())
        else:
            import glob
            parts = sorted(glob.glob(os.path.join(parquet_dir, "part-*.parquet")))
            if parts:
                df = pd.read_parquet(parts[-1])
                print("Dataset label distribution (last chunk):")
                print(df["emotion"].value_counts())
    except Exception:
        pass

if __name__ == "__main__":
    main()
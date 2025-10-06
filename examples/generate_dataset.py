"""
Generate a labeled emotion dataset from raw texts using a local LLM (Stable LM 2 1.6B via Ollama).

Reads all text files from a data directory, splits them into phrases, classifies each phrase into one
of the 8 EmoTiny labels using Ollama, and saves the resulting dataset to Parquet and optionally CSV.
"""

import os
import re
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import requests
from tqdm import tqdm

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "mischievous", "love"]

def read_text_files(data_dir: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
    texts: List[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in extensions:
                path = os.path.join(root, name)
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


def normalize_label(raw: str) -> str:
    t = raw.strip().lower().strip("'\"` .!?")
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


def classify_phrase_with_ollama(phrase: str, model: str = "stablelm2:1.6b", host: str = "http://localhost:11434", temperature: float = 0.0, timeout: int = 60) -> str:
    prompt = (
        "You are an emotion classifier for short text in English and Portuguese.\n"
        "Classify the emotion of the following phrase into EXACTLY ONE of these labels:\n"
        "neutral, happy, sad, angry, surprised, disgusted, mischievous, love.\n\n"
        "Use these definitions:\n"
        "- neutral: neutral/calm state, phrase without strong emotion\n"
        "- happy: joy, happiness, very positive emotion\n"
        "- sad: sadness, melancholy, feeling down\n"
        "- angry: anger, frustration, very negative emotion\n"
        "- surprised: surprise, shock, fear, unexpected event\n"
        "- disgusted: disgust, revulsion, aversion\n"
        "- mischievous: playful, sassy, sexy, seductive\n"
        "- love: love, explicit affection, romantic\n\n"
        "Return ONLY the label word (no punctuation, no explanation). "
        "Use English labels even for Portuguese text.\n\n"
        f"Phrase: {phrase}\nLabel:"
    )
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return normalize_label(data.get("response", ""))


def main():
    parser = argparse.ArgumentParser(description="Generate emotion dataset using Ollama Stable LM 2 1.6B")
    parser.add_argument("--data-dir", default="./data", help="Directory containing raw text files")
    parser.add_argument("--output-parquet", default="./data/emotions.parquet", help="Output Parquet file path")
    parser.add_argument("--output-csv", default="./data/emotions.csv", help="Optional output CSV file path")
    parser.add_argument("--model", default="stablelm2:1.6b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server host")
    parser.add_argument("--min-words", type=int, default=2, help="Minimum words per phrase")
    parser.add_argument("--max-chars", type=int, default=400, help="Maximum characters per phrase")
    parser.add_argument("--max-phrases", type=int, default=0, help="Maximum number of phrases to label (0 = no limit)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()
    texts = read_text_files(args.data_dir)
    if not texts:
        print(f"No text files found in {args.data_dir}")
        return
    all_phrases: List[str] = []
    for t in texts:
        phrases = split_into_phrases(t, min_words=args.min_words, max_chars=args.max_chars)
        all_phrases.extend(phrases)
    if args.max_phrases and args.max_phrases > 0:
        all_phrases = all_phrases[: args.max_phrases]
    print(f"Total phrases to classify: {len(all_phrases)}")
    cache: Dict[str, str] = {}
    labels: List[str] = []
    for p in tqdm(all_phrases, desc="Classifying", unit="phrase"):
        if p in cache:
            labels.append(cache[p])
            continue
        lab = classify_phrase_with_ollama(p, model=args.model, host=args.host)
        cache[p] = lab
        labels.append(lab)
    df = pd.DataFrame({"text": all_phrases, "emotion": labels})
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    if os.path.exists(args.output_parquet) and not args.overwrite:
        print(f"Output Parquet exists: {args.output_parquet}. Use --overwrite to replace.")
    else:
        df.to_parquet(args.output_parquet, index=False)
        print(f"Saved Parquet: {args.output_parquet}")
    if args.output_csv:
        if os.path.exists(args.output_csv) and not args.overwrite:
            print(f"Output CSV exists: {args.output_csv}. Use --overwrite to replace.")
        else:
            df.to_csv(args.output_csv, index=False)
            print(f"Saved CSV: {args.output_csv}")
    print("Dataset label distribution:")
    print(df["emotion"].value_counts())

if __name__ == "__main__":
    main()
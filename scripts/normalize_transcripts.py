#!/usr/bin/env python3
"""Convert WhisperX .txt transcripts into corrected Traditional Chinese Markdown.

The script uses GitHub Models from GitHub Actions. Existing .md files are left
untouched so manually reviewed files are never overwritten.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://models.github.ai/inference/chat/completions"
MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))

SYSTEM_PROMPT = r"""
你是一位嚴謹的繁體中文編輯，正在整理由 WhisperX 產生的網路短影音逐字稿。

工作要求：
1. 將內容轉為繁體中文，使用臺灣常用字詞與標點。
2. 校正明顯的語音辨識錯誤、同音錯字及斷句錯誤，但不得臆造原文沒有的事實、案例或主張。
3. 保留講者原本的論點、語氣與敘事順序；可刪除無意義的口頭贅詞和重複句，但不可改變立場。
4. 整理成易讀的 Markdown：第一行是 H1 標題，正文分成自然段落，並依內容加入少量、有意義的 H2 小節。
5. 不要加入摘要、評論、查證結果、編者按、資料來源，也不要寫「以下是整理結果」之類的說明。
6. 不要使用 Markdown 程式碼區塊。
7. 檔案名稱必須原樣作為 JSON 的 key；Markdown 內的標題則轉成繁體中文並補上必要標點。

你會收到一到數篇文章。請只回傳一個合法 JSON 物件：key 是完整原始檔名，value 是整理完成的 Markdown 字串。不要在 JSON 前後加入任何文字。
""".strip()


def chunks(items: List[Path], size: int) -> Iterable[List[Path]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def build_user_prompt(paths: List[Path]) -> str:
    parts: List[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        parts.append(f"=== FILE: {path.name} ===\n{text}")
    return "\n\n".join(parts)


def extract_json(text: str) -> Dict[str, str]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return {str(k): str(v) for k, v in parsed.items()}


def request_model(paths: List[Path]) -> Dict[str, str]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(paths)},
        ],
        "temperature": 0.15,
        "max_tokens": 12000,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            API_URL,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "Wan-Short-transcript-normalizer",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return extract_json(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail[:1000]}")
            retry_after = exc.headers.get("Retry-After")
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last_error
            delay = float(retry_after) if retry_after else min(60, 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            delay = min(60, 2 ** attempt)
        print(f"Retrying batch after error: {last_error}", file=sys.stderr)
        time.sleep(delay)

    raise RuntimeError(f"Model request failed after retries: {last_error}")


def clean_markdown(markdown: str, source: Path) -> str:
    text = markdown.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if not text.startswith("# "):
        title = source.stem
        text = f"# {title}\n\n{text}"
    return text.rstrip() + "\n"


def process_batch(paths: List[Path]) -> None:
    try:
        results = request_model(paths)
        missing = [path for path in paths if path.name not in results]
        if missing:
            raise ValueError("Missing keys: " + ", ".join(path.name for path in missing))
        for path in paths:
            output = path.with_suffix(".md")
            output.write_text(clean_markdown(results[path.name], path), encoding="utf-8")
            print(f"Created {output.relative_to(ROOT)}")
    except Exception:
        if len(paths) == 1:
            raise
        # Retry files individually if a multi-file JSON response is malformed or incomplete.
        for path in paths:
            process_batch([path])


def main() -> int:
    if not TOKEN:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2

    sources = sorted(
        path for path in ROOT.rglob("*.txt")
        if path.is_file()
        and ".git" not in path.parts
        and not path.with_suffix(".md").exists()
    )
    print(f"Found {len(sources)} transcripts without Markdown counterparts")

    failures: List[str] = []
    for batch in chunks(sources, BATCH_SIZE):
        try:
            process_batch(batch)
        except Exception as exc:
            names = ", ".join(path.name for path in batch)
            failures.append(f"{names}: {exc}")
            print(f"FAILED: {names}: {exc}", file=sys.stderr)
        time.sleep(1.0)

    if failures:
        failure_path = ROOT / "transcript-normalization-failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{len(failures)} batch(es) failed", file=sys.stderr)
        return 1

    stale_failure = ROOT / "transcript-normalization-failures.txt"
    if stale_failure.exists():
        stale_failure.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

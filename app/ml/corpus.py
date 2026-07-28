"""Prompt-engineering hujjatlarini o'qish va bo'laklarga (chunk) ajratish.

Har bir hujjat `##` sarlavhalari bo'yicha bo'linadi. Katta bo'limlar `###`
bo'yicha yana bo'linadi, chunki TF-IDF uzun matnda aniqlikni yo'qotadi —
bir bo'lim ichida bir nechta mavzu bo'lsa, so'rovga mos kelmagan matn ham
tortib kelinadi.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from app.config import CORPUS_DIR

# Bir chunk shu belgidan uzun bo'lsa, ### bo'yicha yana bo'linadi.
_MAX_CHUNK_CHARS = 2600
# Bundan qisqa bo'lak mustaqil ma'no bermaydi — oldingisiga qo'shiladi.
_MIN_CHUNK_CHARS = 180


@dataclass
class Chunk:
    """Korpusning bitta qidiriladigan bo'lagi."""

    chunk_id: str
    doc_id: str
    family: str          # claude | openai
    model: str           # opus-5 | sonnet-5 | fable-5 | opus-4-8 | all
    doc_title: str
    section: str         # sarlavhalar zanjiri: "Tool use > Tool usage"
    text: str
    source: str          # asl URL

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def searchable(self) -> str:
        """Indekslanadigan matn — sarlavha ikki marta, og'irligi oshsin."""
        return f"{self.doc_title} {self.section} {self.section} {self.text}"


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """`---` bilan o'ralgan YAML-simon boshni ajratadi (kutubxonasiz)."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    head = raw[3:end].strip()
    body = raw[end + 4 :].lstrip("\n")
    meta: dict = {}
    for line in head.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


def _split_by_heading(text: str, level: int) -> list[tuple[str, str]]:
    """Matnni berilgan darajadagi sarlavhalar bo'yicha (sarlavha, tana) ga bo'ladi.

    Kod bloklari ichidagi `#` sarlavha deb qabul qilinmasligi kerak —
    ```bash ichidagi izohlar aynan shunday boshlanadi.
    """
    marker = "#" * level + " "
    parts: list[tuple[str, str]] = []
    current_title = ""
    buf: list[str] = []
    in_fence = False

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith(marker):
            if buf or current_title:
                parts.append((current_title, "\n".join(buf).strip()))
            current_title = line[len(marker) :].strip()
            buf = []
        else:
            buf.append(line)

    if buf or current_title:
        parts.append((current_title, "\n".join(buf).strip()))
    return parts


def _chunk_document(meta: dict, body: str) -> Iterator[Chunk]:
    doc_id = meta.get("id", "unknown")
    family = meta.get("family", "unknown")
    model = meta.get("model", "all")
    title = meta.get("title", doc_id)
    source = meta.get("source", "")

    idx = 0
    for h2, h2_body in _split_by_heading(body, 2):
        if not h2_body.strip() and not h2:
            continue

        # Bo'lim kichik bo'lsa — butunligicha bitta chunk.
        if len(h2_body) <= _MAX_CHUNK_CHARS:
            candidates = [(h2 or title, h2_body)]
        else:
            subs = _split_by_heading(h2_body, 3)
            candidates = []
            for h3, h3_body in subs:
                section = f"{h2} > {h3}" if h3 and h2 else (h3 or h2 or title)
                candidates.append((section, h3_body))

        for section, chunk_text in candidates:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < _MIN_CHUNK_CHARS:
                continue
            idx += 1
            yield Chunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                family=family,
                model=model,
                doc_title=title,
                section=section,
                text=chunk_text,
                source=source,
            )


def load_chunks(corpus_dir: Path | None = None) -> list[Chunk]:
    """Korpusdagi barcha .md fayllarni o'qib, chunk ro'yxatini qaytaradi."""
    directory = corpus_dir or CORPUS_DIR
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        meta.setdefault("id", path.stem)
        chunks.extend(_chunk_document(meta, body))
    return chunks


def load_manifest(corpus_dir: Path | None = None) -> list[dict]:
    directory = corpus_dir or CORPUS_DIR
    path = directory / "manifest.json"
    if not path.exists():
        return []
    # PowerShell `Out-File -Encoding utf8` BOM qo'shadi — utf-8-sig uni yeydi.
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else [data]


def strip_markdown(text: str) -> str:
    """PDF va ko'chirma uchun markdown belgilarini tozalaydi."""
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # havolalar
    text = re.sub(r"<[^>]+>", "", text)                      # MDX teglari
    text = re.sub(r"[*_`]{1,3}", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

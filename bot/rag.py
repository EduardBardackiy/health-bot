"""Мини-RAG по библиотеке разборов: chunking + keyword-поиск.

MVP без эмбеддингов: на вопрос ищем релевантные фрагменты по ключевым
словам (уникальные слова вопроса с учётом попаданий) и отдаём топ-фрагменты
в системный промпт. Файлы берутся из FILES_BASE (доки + reports, *.md).
"""
import glob
import os
import re

# Фрагмент ~800 знаков — комфортно и для поиска, и для токенов
CHUNK_LEN = 800
TOP_K = 2
MIN_SCORE = 2  # минимум ключевых слов вопроса, найденных во фрагменте

_tokens = re.compile(r"[а-яёa-z]{3,}")
_STOPWORDS = set(
    "и в не на с что как по а но для от до при этом его её из же под над или "
    "уже если это потом ещё мне ты мы вы они ваш наш весь свой есть был "
    "быть чем так тоже там вот всё можно нужно когда очень без через "
    "the for and with from this that which are you your can will not "
    "have has what when where how why who out get about into over".split()
)


def _split(text: str) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 <= CHUNK_LEN:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para[:CHUNK_LEN]
    if buf:
        chunks.append(buf)
    return chunks


def build_index(base: str) -> list[dict]:
    """Индекс всех *.md из docs/ и reports/ внутри base."""
    index: list[dict] = []
    for sub in ("reports", "docs"):
        for path in glob.glob(os.path.join(base, sub, "*.md")):
            title = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            if not text.strip():
                continue
            index.extend({"source": title, "text": chunk} for chunk in _split(text))
    return index


def search(question: str, index: list[dict]) -> str:
    """Релевантные фрагменты одним текстом (или пустая строка)."""
    tokens = set(_tokens.findall(question.lower())) - _STOPWORDS
    if not tokens or not index:
        return ""
    scored = []
    for chunk in index:
        text_lower = chunk["text"].lower()
        hits = sum(1 for t in tokens if t in text_lower)
        if hits:
            scored.append((hits, chunk))
    scored.sort(key=lambda pair: -pair[0])

    out: list[str] = []
    total = 0
    for hits, chunk in scored[:TOP_K]:
        if hits < MIN_SCORE:
            break
        if total + len(chunk["text"]) > 3500:
            break
        out.append(f"### Разбор «{chunk['source']}»\n{chunk['text']}")
        total += len(chunk["text"])
    return "\n\n---\n\n".join(out)

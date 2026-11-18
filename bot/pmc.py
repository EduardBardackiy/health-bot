"""Пред-поиск PubMed-статей через Europe PMC REST API (бесплатно, без ключа).

Гибрид «пред-поиск + отложенная валидация»: топ-N статей по вопросу уходит
в системный промпт — модель ссылается на реальные PMID, а не выдумывает;
валидация PMID из ответов — вторым этапом (пока отложена).

Используется EBI-зеркало (www.ebi.ac.uk) — основной домен europepmc.org
блокирует часть IP (проверено 18.11.2026). Токенизация реюзается из rag.py.

Medline кириллицу не находит (HitCount 0), поэтому русский вопрос сначала
переводится в 3-6 английских терминов лёгким вызовом LLM
(llm.get_search_terms); если среди токенов вопроса есть латинские —
ищем по ним без вызова LLM.
"""
import json
import logging
import re
import time
import urllib.parse
import urllib.request

from rag import _STOPWORDS, _tokens

import llm

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
TOP_N = 4
CACHE_TTL = 3600  # секунд (~1 час)
ABSTRACT_MAX = 600  # знаков аннотации на статью
TOTAL_MAX = 4000  # общий лимит текста в промпт
QUERY_TOKENS_MAX = 8  # англ. терминов в запросе, не больше
TIMEOUT = 15  # секунд на HTTP-запрос

logger = logging.getLogger("pmc")

_cache: dict[str, tuple[float, str]] = {}  # вопрос -> (время, результат)


def search(question: str) -> str:
    """Топ-N статей по вопросу одним текстом (или пустая строка).

    Ошибки (сеть, LLM) логируются и не бросаются наружу: пустой результат —
    «подсказок нет», чат с моделью работает как раньше.
    """
    q = question.strip().lower()
    if not q:
        return ""
    hit = _cache.get(q)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    tokens = [t for t in _tokens.findall(q) if t not in _STOPWORDS]
    latin = list(dict.fromkeys(t for t in tokens if t.isascii()))[:QUERY_TOKENS_MAX]
    try:
        terms = latin or llm.get_search_terms(question)
    except Exception as exc:
        logger.warning("Europe PMC: не получили англ. термины для «%s»: %s", q, exc)
        return ""
    if not terms:
        _cache[q] = (time.time(), "")  # «не медицина» от LLM — кэшируем пусто
        return ""

    try:
        text = _fetch(terms)
    except Exception as exc:  # urllib.HTTPError, URLError, TimeoutError, json...
        logger.warning("Europe PMC: поиск по «%s» не удался: %s", " ".join(terms), exc)
        return ""  # сетевые сбои не кэшируем — ретрай на следующем сообщении
    _cache[q] = (time.time(), text)
    if len(_cache) > 100:  # страховка от роста: сброс старых записей
        for old in sorted(_cache, key=lambda k: _cache[k][0])[:-100]:
            _cache.pop(old, None)
    return text


def _fetch(latin: list[str]) -> str:
    params = urllib.parse.urlencode({
        "query": " AND ".join(latin) + " AND SRC:MED",  # SRC:MED — только Medline
        "format": "json",
        "resultType": "core",
        "pageSize": TOP_N,
    })
    req = urllib.request.Request(
        f"{API}?{params}", headers={"User-Agent": "health-bot/1.0"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    results = (data.get("resultList") or {}).get("result") or []
    if not results:
        return ""

    out = [f"Найдено всего: {data.get('hitCount', '?')}."]
    total = 0
    for i, res in enumerate(results, 1):
        entry = (
            f"{i}. «{res.get('title')}» — {res.get('authorString')}, "
            f"{res.get('journalInfo', {}).get('journal', {}).get('title') if isinstance(res.get('journalInfo'), dict) else res.get('journalInfo')}, "
            f"{res.get('pubYear')}. PMID: {res.get('pmid')}."
        )
        abstract = re.sub(r"<[^>]+>", "", res.get("abstractText") or "")
        if abstract:
            entry += f"\n   {abstract[:ABSTRACT_MAX].strip()}…"
        if total + len(entry) > TOTAL_MAX:
            break
        out.append(entry)
        total += len(entry)
    return "\n".join(out)

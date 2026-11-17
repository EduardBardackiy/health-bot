"""Вызов LLM с выбором провайдера: Amvera LLM Inference или OpenAI API.

Провайдер по умолчанию задаётся переменной окружения LLM_PROVIDER
(amvera|openai), его можно менять на лету командой /model в боте.
"""
import os

from openai import OpenAI

PROVIDERS: dict[str, dict] = {
    "amvera": {
        "key_env": "AMVERA_API_TOKEN",
        "base_url": os.getenv("AMVERA_BASE_URL", "https://inference.waw0.amvera.ru/v1"),
        "model": os.getenv("LLM_MODEL", "glm-5.1"),  # LLM_MODEL — модель Amvera
        "max_param": "max_tokens",
        "temperature": 0.4,
        "label": "Amvera LLM Inference",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MODEL", "gpt-5.5"),  # OPENAI_MODEL — модель OpenAI
        "max_param": "max_completion_tokens",  # OpenAI не принимает max_tokens
        "temperature": None,  # модели reasoning поддерживают только дефолт (1)
        "label": "OpenAI API",
    },
}

_clients: dict[str, OpenAI] = {}
_override: str | None = None  # смена провайдера командой /model в рантайме


def _current() -> str:
    """Активный провайдер: ручная команда /model либо LLM_PROVIDER из .env."""
    if _override is not None:
        return _override
    return os.getenv("LLM_PROVIDER", "amvera").lower()


def _get_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """Возвращает (клиент, имя модели) для провайдера (по умолчанию — текущего)."""
    name = provider if provider is not None else _current()
    cfg = PROVIDERS[name]
    client = _clients.get(name)
    if client is None:
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            raise RuntimeError(
                f"Провайдер «{name}»: не задана переменная окружения {cfg['key_env']}"
            )
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        _clients[name] = client
    return client, cfg["model"]


def get_provider() -> str:
    return _current()


def set_provider(name: str) -> None:
    """Переключить провайдер (amvera|openai); кэш клиентов сохраняется."""
    global _override
    name = name.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Неизвестный провайдер: {name}")
    _override = name


def get_reply(history: list[dict]) -> str:
    """history — список {"role": "user"|"assistant", "content": str}.

    Возвращает текст ответа модели текущего провайдера.
    """
    from prompts import SYSTEM_PROMPT

    client, model = _get_client()
    cfg = PROVIDERS[get_provider()]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    params: dict = {cfg["max_param"]: 2000}
    if cfg["temperature"] is not None:
        params["temperature"] = cfg["temperature"]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        **params,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Модель вернула пустой ответ")
    return text


def get_analysis_reply(
    text: str | None = None,
    image_b64: str | None = None,
    mime: str = "image/jpeg",
) -> str:
    """Разбор лабораторных анализов (отдельный промпт, вне истории диалога).

    image_b64 — фото бланка (base64): запрос всегда через провайдера «openai»
    (у Amvera glm vision не проверен). Только текст (PDF) — текущим провайдером.
    """
    from prompts import ANALYSIS_PROMPT

    provider = "openai" if image_b64 else _current()
    client, model = _get_client(provider)
    cfg = PROVIDERS[provider]
    if image_b64:
        content: str | list = [
            {"type": "text", "text": text or "Разбери бланк анализов."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]
    else:
        content = text or ""
    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": content},
    ]
    params: dict = {cfg["max_param"]: 3000}
    if cfg["temperature"] is not None:
        params["temperature"] = cfg["temperature"]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        **params,
    )
    reply = (resp.choices[0].message.content or "").strip()
    if not reply:
        raise RuntimeError("Модель вернула пустой ответ")
    return reply

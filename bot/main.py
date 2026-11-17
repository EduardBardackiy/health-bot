"""Telegram-бот «Health-консультант» (проект Health_amvera на Amvera)."""
import asyncio
import base64
import json
import logging
import mimetypes
import os
import socket
import sys
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv
from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import PROVIDERS, get_analysis_reply, get_provider, get_reply, set_provider  # noqa: E402
from prompts import SYSTEM_PROMPT  # noqa: E402

HISTORY_LIMIT = 12  # сообщений диалога, пересылаемых в модель
TG_LIMIT = 4096
PDF_MAX_PAGES = 10
PDF_MAX_CHARS = 8000

# Проектная документация, доступная боту
DOCS_MENU = {
    "Цитруллин малат 8 г — разбор": "reports/2026-11-05-citrulline-malate-8g-strength-evidence.md",
    "Оксандролон vs Стромбафорт": "docs/2026-10-30 — ААС — оксандролон vs стромбафорт, разбор курса — конспект.md",
    "25 анализов перед курсом или ГЗТ": "docs/25 АНАЛИЗОВ ПЕРЕД КУРСОМ ИЛИ ГЗТ..md",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bot")

# История диалогов персистентна: на Amvera монтируется /data,
# локально — data/ в корне проекта.
DATA_DIR = "/data" if os.path.isdir("/data") else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
HISTORIES_FILE = os.path.join(DATA_DIR, "histories.json")

dp = Dispatcher()
# История диалогов: chat_id -> [{"role": ..., "content": ...}, ...]
histories: dict[int, list[dict]] = {}


def load_histories() -> dict[int, list[dict]]:
    if not os.path.exists(HISTORIES_FILE):
        return {}
    try:
        with open(HISTORIES_FILE, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("Не удалось загрузить истории из %s: %s", HISTORIES_FILE, exc)
        return {}


def save_histories() -> None:
    """Атомарная запись: временный файл + rename."""
    tmp = HISTORIES_FILE + ".tmp"
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False)
        os.replace(tmp, HISTORIES_FILE)
    except OSError as exc:
        logger.warning("Не удалось сохранить истории в %s: %s", HISTORIES_FILE, exc)


def chunk(text: str, size: int = TG_LIMIT) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def get_history(chat_id: int) -> list[dict]:
    return histories.setdefault(chat_id, [])


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я Health-консультант — бот проекта Health_amvera.\n\n"
        "Задавай вопросы по здоровью, тренировкам, добавкам и фармакологии. "
        "Отвечаю по подтверждённым источникам и отделяю доказанные факты "
        "от личного опыта авторов.\n\n"
        "Команды:\n"
        "/help — что я умею\n/report — список разборов проекта\n"
        "/model — посмотреть/сменить LLM-провайдера\n"
        "/clear — сбросить историю диалога\n\n"
        "Фото бланка анализов или PDF с результатами пришли прямо в чат — "
        "разберу таблицей."
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Что я умею:\n\n"
        "• Отвечаю на вопросы по здоровью, тренировкам, добавкам и фармакологии — "
        "с акцентом на доказательность и источники (PubMed).\n"
        "• /report — прислать разбор из библиотеки проекта.\n"
        "• Разбор анализов: фото бланка или PDF с результатами — таблицей, "
        "с нормами и статусами.\n"
        "• /model [/амвера|openai] — LLM-провайдер.\n"
        "• /clear — очистить контекст диалога.\n\n"
        "Я не врач: при серьёзных симптомах — к специалисту."
    )


@dp.message(Command("report"))
async def cmd_report(message: Message) -> None:
    lines = ["Библиотека разборов проекта — пришли цифру получателю:"]
    for i, title in enumerate(DOCS_MENU, 1):
        lines.append(f"{i}. {title}")
    await message.answer("\n".join(lines))


@dp.message(F.text.regexp(r"^/doc(\d)$"))
async def cmd_doc(message: Message) -> None:
    idx = int(message.text.strip()[4:]) - 1
    titles = list(DOCS_MENU)
    if not (0 <= idx < len(titles)):
        await message.answer("Нет такого номера. Посмотри /report.")
        return
    title = titles[idx]
    path = DOCS_MENU[title]
    if os.path.exists(path):
        await message.answer_document(FSInputFile(path), caption=title)
    else:
        await message.answer(f"Файл «{title}» не найден на сервере.")
    return


@dp.message(Command("model"))
async def cmd_model(message: Message) -> None:
    arg = (message.text or "").replace("/model", "").strip().lower()
    if not arg:
        cfg = PROVIDERS[get_provider()]
        await message.answer(
            f"Текущий провайдер: {get_provider()} ({cfg['label']}), модель «{cfg['model']}».\n"
            f"Переключить: /model amvera или /model openai"
        )
        return
    try:
        set_provider(arg)
        cfg = PROVIDERS[arg]
        await message.answer(f"Провайдер переключён: {arg} ({cfg['label']}), модель «{cfg['model']}».")
    except ValueError:
        await message.answer(
            "Нет такого провайдера. Доступны: " + ", ".join(PROVIDERS)
        )


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    histories.pop(message.chat.id, None)
    save_histories()
    await message.answer("История диалога сброшена.")


def _pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    parts = [page.extract_text() or "" for page in reader.pages[:PDF_MAX_PAGES]]
    text = "\n".join(parts).strip()
    if len(text) < 50:
        raise ValueError(
            "В PDF нет текстового слоя (похоже на скан) — пришли бланк фото постранично"
        )
    return text[:PDF_MAX_CHARS]


async def _analyze(message: Message, file_id: str, mime: str, is_pdf: bool) -> None:
    """Разбор бланка анализов: фото → vision, PDF → текстовый слой."""
    await message.bot.send_chat_action(message.chat.id, "typing")
    buf = BytesIO()
    file = await message.bot.get_file(file_id)
    await message.bot.download_file(file.file_path, destination=buf)
    try:
        if is_pdf:
            reply = await asyncio.to_thread(
                get_analysis_reply, text=_pdf_text(buf.getvalue())
            )
        else:
            reply = await asyncio.to_thread(
                get_analysis_reply, image_b64=base64.b64encode(buf.getvalue()).decode(),
                mime=mime,
            )
    except Exception as exc:
        logger.exception("Ошибка разбора анализов")
        err = exc.__cause__ or exc
        await message.answer(f"Ошибка разбора: {err}")
        return
    for part in chunk(reply):
        await message.answer(part)


@dp.message(F.photo)
async def on_photo(message: Message) -> None:
    if not message.photo:
        return
    await _analyze(message, message.photo[-1].file_id, "image/jpeg", is_pdf=False)


@dp.message(F.document)
async def on_document(message: Message) -> None:
    doc = message.document
    mime = doc.mime_type or mimetypes.guess_type(doc.file_name or "")[0] or ""
    name = (doc.file_name or "").lower()
    is_pdf = mime == "application/pdf" or name.endswith(".pdf")
    if not (is_pdf or mime.startswith("image/")):
        await message.answer(
            "Поддерживаю фото бланка (можно как документ, без сжатия) "
            "или PDF с результатами анализов."
        )
        return
    await _analyze(message, doc.file_id, mime or "application/octet-stream", is_pdf)


@dp.message(F.text)
async def on_message(message: Message) -> None:
    if not message.text:
        return
    history = get_history(message.chat.id)
    history.append({"role": "user", "content": message.text})
    try:
        reply = get_reply(history[-HISTORY_LIMIT * 2 :])
    except Exception as exc:
        logger.exception("Ошибка LLM")
        history.pop()  # неудавшийся ход не оставляем в истории
        err = exc.__cause__ or exc
        await message.answer(f"Ошибка: {err}")
        return
    history.append({"role": "assistant", "content": reply})
    # Храним не больше последних HISTORY_LIMIT пар «вопрос-ответ»
    del history[: -HISTORY_LIMIT * 2]
    save_histories()
    for part in chunk(reply):
        await message.answer(part)


def pin_telegram_ip() -> None:
    """Через VPN маршрутизируется не весь пул IP Telegram.

    Если DNS выдаёт нерабочий IP — пиним рабочую ноду через TG_PIN_IP
    (подменяет getaddrinfo только для api.telegram.org; SNI/TLS сохраняются).
    В Docker/у хостинге переменную просто не задаём.
    """
    pin = os.getenv("TG_PIN_IP")
    if not pin:
        return
    orig = socket.getaddrinfo

    def pinned(host, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if host == "api.telegram.org":
            host = pin
        return orig(host, *args, **kwargs)

    socket.getaddrinfo = pinned
    logger.info("api.telegram.org pinned to %s", pin)


async def main() -> None:
    load_dotenv()
    pin_telegram_ip()
    histories.update(load_histories())
    logger.info("Загружено диалогов из %s: %d", HISTORIES_FILE, len(histories))
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("Не задана переменная окружения BOT_TOKEN")
    # Прокси для Telegram (api.telegram.org заблокирован в РФ):
    # http://host:port или socks5://host:port. Прокси отдаём и LLM-клиенту (httpx)
    # через переменные окружения, т.к. httpx их читает сам.
    proxy = os.getenv("PROXY_URL")
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
    bot = Bot(token=token, session=AiohttpSession(proxy=proxy or None))
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

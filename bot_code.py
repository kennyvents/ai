import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# ===== ваши импорты обработки =====
# Убедитесь, что каталог с папкой utils есть в PYTHONPATH (или запускайте бот из корня проекта)
from utils.data_utils import read_data, normalize_data, get_result
from utils.match_finder import find_top_matches
from utils.excel_log import get_top_ent

# ---------- базовая настройка ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("excel-bot")

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Не найден BOT_TOKEN в .env")

# Разрешённые типы Excel
EXCEL_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
}
EXCEL_EXTS = {"xlsx", "xls"}

# Пути проекта
FILES_DIR = "files"
RESULTS_DIR = os.path.join(FILES_DIR, "results")
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

INCOMING_XLSX = os.path.join(FILES_DIR, "заявка.xlsx")
FINAL_XLSX = os.path.join(RESULTS_DIR, "finally_results.xlsx")  # итоговый файл, который отправим


# ---------- синхронная функция обработки ----------
def process_excel() -> str:
    """
    Запускает ваш пайплайн обработки.
    Возвращает путь к итоговому файлу (FINAL_XLSX).
    Ожидается, что read_data('modified_prices', 'заявка') прочитает:
      - prices из 'files/modified_prices.xlsx'
      - заявку из 'files/заявка.xlsx'
    При необходимости адаптируйте реализацию read_data / аргументы.
    """
    # 1) читаем данные
    data = read_data('modified_prices', 'заявка')   # <-- оставлено как в вашем коде
    # 2) нормализация
    data = normalize_data(data[0], data[1])
    # 3) поиск, результат, сводка
    result_data = find_top_matches(data[0], data[1], data[2])
    get_result(result_data)
    # get_top_ent, судя по сигнатуре, кладёт .xlsx в указанный каталог
    # Пример ожидаемых путей: 'files/results/emb_results' и 'files/results/finally_results'
    get_top_ent('files/results/emb_results', 'files/results/finally_results')

    # Гарантируем ожидаемое имя итогового файла
    # Если get_top_ent создаёт название иначе — переименуем под FINAL_XLSX
    # Если файл уже так и называется — трогать не нужно.
    if not os.path.exists(FINAL_XLSX):
        # Попробуем найти единственный .xlsx в целевой папке и принять его за итог
        generated = [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".xlsx")]
        if len(generated) == 1:
            src = os.path.join(RESULTS_DIR, generated[0])
            if src != FINAL_XLSX:
                os.replace(src, FINAL_XLSX)
        elif len(generated) > 1:
            # Берём самый свежий
            latest = max(
                (os.path.join(RESULTS_DIR, f) for f in generated),
                key=lambda p: os.path.getmtime(p)
            )
            os.replace(latest, FINAL_XLSX)

    if not os.path.exists(FINAL_XLSX):
        raise FileNotFoundError(
            "Обработка завершилась, но итоговый файл не найден. "
            "Проверьте, что get_top_ent пишет .xlsx в 'files/results/finally_results.xlsx' "
            "или адаптируйте логику выше."
        )
    return FINAL_XLSX


# ---------- handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 Привет! Пришлите Excel с заявкой(.xlsx/.xls).\n"
        "А я верну подходящие позиции из нашей базы "
        "P.S пока поддерживает только поиск воздуховодов, всё остальное будет не корректно"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Принимает Excel, сохраняет как 'files/заявка.xlsx', запускает обработку и шлёт 'finally_results.xlsx'."""
    print(update.message.from_user.name)
    doc = update.message.document
    if not doc:
        return

    filename = doc.file_name or "file"
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    is_excel = (ext in EXCEL_EXTS) or (doc.mime_type in EXCEL_MIME_TYPES if doc.mime_type else False)
    if not is_excel:
        await update.message.reply_text("⚠️ Это не похоже на Excel. Пришлите .xlsx или .xls.")
        return

    # 1) сохраняем входящий файл строго под INCOMING_XLSX
    file = await doc.get_file()
    await file.download_to_drive(custom_path=INCOMING_XLSX)
    log.info("Входящий файл сохранён: %s", INCOMING_XLSX)
    await update.message.reply_text("📥 Файл получен. Запускаю обработку...")

    # 2) запускаем обработку в отдельном потоке (не блокируем asyncio loop)
    try:
        result_path = await asyncio.to_thread(process_excel)
    except Exception as e:
        log.exception("Ошибка обработки:")
        await update.message.reply_text(f"❌ Ошибка обработки: {e}")
        return

    # 3) отправляем итоговый файл
    try:
        with open(result_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=os.path.basename(result_path),
                caption="✅ Готово: итоговый файл."
            )
    except FileNotFoundError:
        await update.message.reply_text("❌ Не нашёл итоговый файл после обработки.")
    except Exception as e:
        log.exception("Ошибка отправки результата:")
        await update.message.reply_text(f"❌ Не удалось отправить результат: {e}")


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пришлите Excel, я его обработаю и верну результат 🙂")


# ---------- точка входа ----------
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(~filters.Document.ALL, unknown))

    log.info("Бот запущен.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

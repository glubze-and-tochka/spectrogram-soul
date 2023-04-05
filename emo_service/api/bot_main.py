from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
from bot.filters import register_all_filters  # noqa
from bot.handlers import register_all_handlers  # noqa
from bot.settings import BotSettings  # noqa


async def __on_start_up(dp: Dispatcher) -> None:
    register_all_filters(dp)
    register_all_handlers(dp)


def start_bot(settings: BotSettings = BotSettings()):  # noqa
    """Factory function for start aiogram bot."""
    bot = Bot(token=settings.API_TOKEN, parse_mode="HTML")  # type: ignore
    dp = Dispatcher(bot, storage=MemoryStorage())
    executor.start_polling(dp, skip_updates=True, on_startup=__on_start_up)

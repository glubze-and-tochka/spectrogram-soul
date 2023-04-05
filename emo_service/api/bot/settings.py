from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv  # noqa

load_dotenv()


@dataclass
class BotSettings:
    """Environment settings for bot."""

    API_TOKEN: str | None = getenv("AIOGRAM_API_TOKEN")  # type: ignore

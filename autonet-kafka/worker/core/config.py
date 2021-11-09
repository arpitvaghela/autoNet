import logging
import sys

from core.logging import InterceptHandler
from loguru import logger
from starlette.config import Config

config = Config(".env")


PROJECT_NAME: str = config("PROJECT_NAME", default="autoNET-controller")
KAFKA_URI: str = config("KAFKA_HOST")
KAFKA_PORT: str = config("KAFKA_PORT")
KAFKA_INSTANCE = KAFKA_URI + ":" + KAFKA_PORT
DEBUG: bool = config("DEBUG", cast=bool, default=False)

LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO

logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

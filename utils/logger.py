import logging
import os
import sys

_DEFAULT_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_ENABLE_COLOR = os.environ.get("ANY3D_LOG_COLOR", "").lower() in ("1", "true", "yes", "on")
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class _ColorFormatter(logging.Formatter):
    _RESET = "\x1b[0m"
    _COLORS = {
        logging.CRITICAL: "\x1b[1;31m",
        logging.ERROR: "\x1b[31m",
        logging.WARNING: "\x1b[33m",
        logging.INFO: "\x1b[36m",
        logging.DEBUG: "\x1b[35m",
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self._COLORS.get(record.levelno)
        if color:
            return f"{color}{msg}{self._RESET}"
        return msg


def setup_logging(level: str | int = None) -> None:
    if level is None:
        level = _DEFAULT_LEVEL
    if isinstance(level, str):
        level = logging.getLevelName(level)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        use_color = _ENABLE_COLOR or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
        formatter = _ColorFormatter(_FORMAT) if use_color else logging.Formatter(_FORMAT)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"Any3D.{name}")

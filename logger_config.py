# logger_config.py
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": "errors.log",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
        # 你甚至可以单独关闭某个啰嗦库的日志
        "urllib3": {"level": "WARNING"},
    },
}


def setup():
    logging.config.dictConfig(LOGGING_CONFIG)

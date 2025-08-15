import logging
import sys

from logging.handlers import RotatingFileHandler
from pathlib import Path

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s')
CRITICAL_FORMATTER = logging.Formatter('[%(levelname)-8s] [%(asctime)s] - %(name)s - %(message)s')


class LessThanFilter(logging.Filter):
    def __init__(self, exclusive_maximum, name=""):
        super(LessThanFilter, self).__init__(name)
        self.max_level = exclusive_maximum

    def filter(self, record):
        # non-zero return means we log this message
        return 1 if record.levelno < self.max_level else 0


def log_to_file(logger: logging.Logger, output_path: str, base_logdir: Path = "logs") -> logging.Logger:
    """
    add an output file to logger
    :param logger:
    :param output_path:
    :param base_logdir:
    """
    base_logdir = Path(base_logdir)
    base_logdir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    assert output_path.suffix == ".log"

    log_file_path = base_logdir / output_path

    file_handler = RotatingFileHandler(
        filename=log_file_path.as_posix(),
        maxBytes=10_000_000,  # 10 MB per log file
        backupCount=5  # Keep 5 old logs: log.log.1, log.log.2, ...
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(log_name=__file__, override_log_function=True, filename=None, notification_level="all"):
    """
    Get a logger interface for printing out a message. All the message would be forwarded
    to its respective standard output pipe.

    For error message. All messages would be sent to a gchat channel, defined in the dot-env
    file. In case there is no value defined in the dot-env file. The channel would be `athena-scoring-e-dev`

    :param log_name:
    :param override_log_function:
    :param filename: output filename
    :param notification_level: Set environment level to emit to Google Chat
    :return:
    """
    logger = logging.getLogger(log_name)

    if override_log_function is True:
        # So the log will also output the message into the console.
        logger.setLevel(logging.INFO)
        stream_handler_info = logging.StreamHandler(stream=sys.stdout)
        stream_handler_info.setLevel(logging.INFO)
        stream_handler_info.setFormatter(formatter)
        stream_handler_info.addFilter(LessThanFilter(logging.ERROR))

        stream_handler_error = logging.StreamHandler(stream=sys.stderr)
        stream_handler_error.setLevel(logging.ERROR)
        stream_handler_error.setFormatter(CRITICAL_FORMATTER)

        stream_handler_critical = logging.StreamHandler(stream=sys.stderr)
        stream_handler_critical.setLevel(logging.CRITICAL)
        stream_handler_critical.setFormatter(CRITICAL_FORMATTER)

        if len(logger.handlers) == 0:
            logger.addHandler(stream_handler_info)
            logger.addHandler(stream_handler_error)
            logger.addHandler(stream_handler_critical)
            logger.propagate = False

        __override_log_function(logger)

    if filename:
        logger = log_to_file(logger, filename)

    return logger


def __override_log_function(logger):
    original_log_info_function = logger.info
    original_log_warn_function = logger.warning
    original_log_error_function = logger.error
    original_log_critical_function = logger.critical

    def info(*args):
        logger.propagate = False
        original_log_info_function(*args)

    def warn(*args):
        logger.propagate = False
        original_log_warn_function(*args)

    def error(*args, **kwargs):
        logger.propagate = False
        original_log_error_function(*args, **kwargs)

    def critical(*args, **kwargs):
        original_log_critical_function(*args, **kwargs)

    logger.info = info
    logger.warning = warn
    logger.error = error
    logger.critical = critical

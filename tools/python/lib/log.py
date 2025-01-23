import datetime
import logging
import os
import sys
from pathlib import Path

import colorama
from global_constants import global_constants


class __ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__colors = {
            logging.WARNING: colorama.Fore.LIGHTYELLOW_EX,
            logging.ERROR: colorama.Fore.LIGHTRED_EX,
            logging.CRITICAL: colorama.Fore.LIGHTRED_EX,
        }

    def format(self, record) -> str:
        message = super().format(record)
        if record.levelno in self.__colors:
            color = self.__colors[record.levelno]
            message = f"{color}{message}{colorama.Style.RESET_ALL}"
        return message


class __ToolbagLogger(logging.Logger):
    def __del__(self):
        self.cleanup()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__canceled = False
        self.__progress_maximum = 0
        self.__progress_minimum = 0
        self.__progress_value = 0

    def cancel(self):
        self.__canceled = True

    def cleanup(self):
        for handler in self.handlers:
            handler.flush()
        self.__canceled = None
        self.__progress_maximum = None
        self.__progress_minimum = None
        self.__progress_value = None

    def hide_progress_bar(self):
        self.__progress_maximum = 0
        self.__progress_minimum = 0
        self.__progress_value = 0
        self.debug(
            "Toolbag Hide Progress Bar",
            extra={
                "toolbag_hide_progress_bar": True,
            },
        )

    def is_canceled(self) -> bool:
        return self.__canceled

    def show_progress_bar(self, minimum: float, maximum: float):
        self.__progress_maximum = maximum
        self.__progress_minimum = minimum
        self.__progress_value = minimum
        self.debug(
            "Toolbag Show Progress Bar",
            extra={
                "toolbag_progress_maximum": maximum,
                "toolbag_progress_minimum": minimum,
                "toolbag_show_progress_bar": True,
            },
        )

    def update_progress_bar(self, delta: float, text: str = ""):
        self.__progress_value += delta
        message = "Toolbag Update Progress Bar"
        value = min(self.__progress_value, self.__progress_maximum)
        value = max(self.__progress_minimum, value)
        if text:
            message += ": " + text
        self.debug(
            message,
            extra={
                "toolbag_progress_text": text,
                "toolbag_progress_value": value,
                "toolbag_update_progress_bar": True,
            },
        )


def __create_log(max_logs: int = 100) -> __ToolbagLogger:
    colorama.init()
    debugs = []
    errors = []
    handlers = [logging.StreamHandler()]
    logs_path = global_constants.tools_path / "_logs"
    os.makedirs(logs_path, exist_ok=True)
    if os.path.isdir(logs_path):
        failed_logs = []
        logs = [logs_path / e for e in os.listdir(logs_path)]
        logs.sort(key=lambda log: log.stat().st_mtime)
        if max_logs > 0:
            while len(failed_logs) + len(logs) >= max_logs and len(failed_logs) < max_logs:
                old_log = logs.pop(0)
                debugs.append(f'Removing log "{os.path.basename(old_log)}"')
                try:
                    os.unlink(old_log)
                except Exception as e:
                    debugs.append(f'Failed to remove "{old_log}"\n{e}')
                    failed_logs.append(old_log)
        now = datetime.datetime.now()
        log_name = Path(sys.executable).stem
        if log_name.lower() == "python" and len(sys.argv) > 0:
            python_file = Path(sys.argv[0])
            if python_file.suffix.lower() == ".py":
                log_name = python_file.stem
            if log_name.lower() == "repo" and len(sys.argv) > 1:
                log_name += f"_{sys.argv[1]}"
        log_path = logs_path / now.strftime(f"{log_name}_%Y-%m-%d_%H_%M_%S.log")
        debugs.append(f'Logging to "{log_path}"')
        handlers.append(logging.FileHandler(log_path))
    elif os.path.exists(logs_path):
        errors.append(f'Logs path "{logs_path}" is not a directory!')
    else:
        errors.append(f'Logs path "{logs_path}" does not exist!')
    handlers[0].setLevel(logging.INFO)
    handlers[0].setFormatter(__ColoredFormatter())
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger_class = logging.setLoggerClass(__ToolbagLogger)
    logger = logging.getLogger("toolbag")
    if isinstance(logger_class, logging.Logger):
        logging.setLoggerClass(logger_class)
    else:
        logging.setLoggerClass(logging.Logger)
    for message in errors:
        logger.error(message)
    for message in debugs:
        logger.debug(message)
    return logger


log = __create_log()

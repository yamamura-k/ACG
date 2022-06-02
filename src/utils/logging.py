from logging import getLogger, StreamHandler, FileHandler

from colorlog import ColoredFormatter


class setupLogger:
    def __init__(self):
        """Return a logger with a ColoredFormatter."""
        self.formatter = ColoredFormatter(
            "%(log_color)s%(levelname)s:%(pathname)s:%(funcName)s(%(lineno)s):%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )

        logger = getLogger()
        self.root_logger = logger

        self.setStreamHandler()
        # self.setFileHandler()

    def setStreamHandler(self):
        handler = StreamHandler()
        handler.setFormatter(self.formatter)
        self.root_logger.addHandler(handler)

    def setFileHandler(self, filename="ObjectDetection.log", log_level=10):
        handler = FileHandler(filename=filename)
        handler.setLevel(log_level)
        # handler.setFormatter(self.formatter)
        self.root_logger.addHandler(handler)

    def setLevel(self, log_level):
        self.root_logger.setLevel(log_level)
        return self.root_logger

    def __call__(self, name):
        return self.root_logger.getChild(name)


setup_logger = setupLogger()

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)            # ← allow DEBUG+ through

format_str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | "
    "%(funcName)s() | %(message)s"
)

# Handlers
stream_handler = logging.StreamHandler()
file_handler   = logging.FileHandler('project.log')

# Set handler-specific levels
stream_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.WARNING)

# Formatter
formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

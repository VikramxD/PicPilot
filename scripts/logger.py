import logging
from logging.handlers import RotatingFileHandler
import os
from rich.logging import RichHandler
from pathlib import Path
from config import LOGS_DIR

# Set up logging configuration (you can put this in a separate module)
logs_dir = Path(LOGS_DIR)
os.makedirs(logs_dir, exist_ok=True)

rich_logger = logging.getLogger(__name__)
rich_logger.setLevel(logging.DEBUG)

# Create a console handler for INFO level and above
console_handler = RichHandler()
console_handler.setLevel(logging.INFO)
rich_logger.addHandler(console_handler)

# Create individual file handlers for each log level
log_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

for level in log_levels:
    log_file = os.path.join(logs_dir, f'app_{logging.getLevelName(level).lower()}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(module)s - %(message)s'))
    rich_logger.addHandler(file_handler)


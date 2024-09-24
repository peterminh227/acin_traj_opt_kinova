import logging
from colorama import init, Fore, Style

try:
    from . import config as cf
except:
    import os 
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as cf


# Initialize colorama
init(autoreset=True)

config = cf.parseArgs()

class CustomFormatter(logging.Formatter):
    """Custom logging formatter with colors."""
    FORMAT = "%(asctime)s   %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.CYAN + FORMAT + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + FORMAT + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + FORMAT + Style.RESET_ALL,
        logging.ERROR: Fore.RED + FORMAT + Style.RESET_ALL,
        logging.CRITICAL: Fore.MAGENTA + FORMAT + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def set_logging_level(level):
    """Set the logging level dynamically."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

# Create a logger object
logger = logging.getLogger(__name__)

# Remove all handlers associated with the root logger object
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create handlers
#file_handler = logging.FileHandler('app.log')
console_handler = logging.StreamHandler()

# Create a custom formatter and set it for handlers
formatter = CustomFormatter()
#file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
#logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set initial logging level
set_logging_level(logging.INFO)

if config.logging_level == "error":
    set_logging_level(logging.ERROR)
elif config.logging_level == "debug":
    set_logging_level(logging.DEBUG)
elif config.logging_level == "warning":
    set_logging_level(logging.WARNING)
elif config.logging_level == "info":
    set_logging_level(logging.INFO)
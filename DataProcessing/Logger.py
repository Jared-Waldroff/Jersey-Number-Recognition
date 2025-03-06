import logging
import sys

class CustomLogger:
    def __init__(self, suppress_logging=False):
        self.suppress_logging = suppress_logging
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        if not self.suppress_logging:
          logging.basicConfig(
              level=logging.INFO,
              format="%(asctime)s [%(levelname)s] %(message)s",
              datefmt="%Y-%m-%d %H:%M:%S",
              stream=sys.stdout
          )
    
    def get_logger(self):
        if self.suppress_logging:
            # Create a dummy logger that does nothing
            logger = logging.getLogger("dummy")
            logger.disabled = True
            return logger
        return logging
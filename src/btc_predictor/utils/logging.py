import sys
import os
import logging
import warnings
from datetime import datetime

class TLogger:
    """Redirects stdout/stderr to both original stream and a file."""
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

def setup_logging(name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Redirect stdout and stderr
    sys.stdout = TLogger(log_file, sys.stdout)
    sys.stderr = TLogger(log_file, sys.stderr)
    
    # Capture python warnings
    # Note: warnings.showwarning redirection
    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        msg = warnings.formatwarning(message, category, filename, lineno, line)
        sys.stderr.write(msg)

    warnings.showwarning = custom_showwarning
    
    print(f"--- Session Started: {timestamp} ---")
    print(f"--- Log File: {log_file} ---")
    
    return log_file

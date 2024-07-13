import datetime

class Logger:
    LOG_LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR"]

    def __init__(self, log_level="DEBUG", display_time=False):
        self.log_level = log_level
        self.display_time = display_time
        self.log(f"Logging level is: {log_level}", "INFO", forced=True)

    def log(self, message, level, forced=False, **kwargs):
        if not forced and not self.is_log_level(level):
            return
        
        msg = f"[{level}] {message}"
        if self.display_time:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"[{timestamp}] {msg}"
        print(msg, **kwargs)

    def is_log_level(self, level):
        try:
            current_level = self.LOG_LEVEL_ORDER.index(level)
            cutoff_level = self.LOG_LEVEL_ORDER.index(self.log_level)
        except ValueError:
            print(f"Invalid Logging Level: {level}")
            return False
        return current_level >= cutoff_level

    def debug(self, message, **kwargs):
        cur_level = "DEBUG"
        self.log(message, cur_level, **kwargs)

    def info(self, message, **kwargs):
        cur_level = "INFO"
        self.log(message, cur_level, **kwargs)

    def warning(self, message, **kwargs):
        cur_level = "WARNING"
        self.log(message, cur_level, **kwargs)

    def error(self, message, **kwargs):
        cur_level = "ERROR"
        self.log(message, cur_level, **kwargs)

if __name__ == "__main__":
    # Example usage:
    logger = Logger(log_level="INFO", display_time=False)
    logger.debug("This is an info message", end="")
    logger.error("This is an error message", end="")

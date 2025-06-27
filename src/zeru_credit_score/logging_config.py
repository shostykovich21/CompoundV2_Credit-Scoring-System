import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to write INFO+ logs (or whatever level you choose)
    to stdout with a consistent timestamped format.
    """
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)
    root.handlers = [handler]
    root.setLevel(level)

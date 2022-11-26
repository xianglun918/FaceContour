import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S'
)

logger = logging
